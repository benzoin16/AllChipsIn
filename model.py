"""
NAFNetSR2x — a NAFNet-style encoder/decoder adapted for this challenge's specific
I/O contract: single-channel input at resolution R, output at resolution 2R,
with joint speckle/Gaussian denoising baked into the same forward pass.

Design choices (see README.md for the reasoning):
  - NAFBlock (SimpleGate + Simplified Channel Attention, no nonlinear activations)
    reimplemented from the NAFNet paper (Chen et al., ECCV 2022) rather than
    imported from the original repo, so we're not dragging in the whole BasicSR
    training framework (which assumes same-resolution restoration and a
    yaml/registry config system that doesn't map cleanly onto "2x SR + denoise
    in one fixed eval script").
  - The network denoises at the LOW-resolution grid, then a single PixelShuffle
    head does the 2x upsample. This keeps compute cheap (all the heavy NAFBlock
    work happens at the smaller resolution) — important given the explicit
    H100 inference-time scoring.
  - Global residual: the network predicts a CORRECTION on top of a cheap
    bicubic upsample of the input, not the image from scratch. This is the
    "residual on a cheap upsample" idea — stabilizes low-frequency content,
    lets the network's capacity focus on noise removal + detail recovery.
  - Output is passed through a hard clamp to [0, 1] so the network cannot
    reproduce the input's out-of-range speckle excursions in its output
    (per the "noise pushes values outside true range" clue in the problem
    statement).
  - Fully convolutional — the same weights handle both 128->256 and 256->512
    since the scale factor is always x2.
  - Post-upsample refinement: a few NAFBlocks run AFTER the PixelShuffle, at
    full output resolution, instead of the upsample head being a single
    conv straight to out_ch. All the encoder/middle/decoder work happens at
    low resolution for speed, but that means the network never gets a
    chance to clean up the newly-created high-res detail — refine_blk_num
    blocks give it that chance, at the cost of full-res compute per block
    (kept small in number since full-res is expensive relative to LR work).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies — replaces GELU/ReLU etc.
    This is the core "nonlinear-activation-free" trick from the paper."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_path=0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        # Simplified Channel Attention: global-avg-pool -> 1x1 conv -> multiply
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg1 = SimpleGate()

        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.sg2 = SimpleGate()

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.drop_path = drop_path

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNetSR2x(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        width=32,
        enc_blk_nums=(2, 2, 4),
        middle_blk_num=6,
        dec_blk_nums=(2, 2, 2),
        refine_blk_num=2,
        aux_noise_map=True,
    ):
        """
        aux_noise_map: if True, a cheap local-variance map is concatenated to
        the input as an extra channel (CBDNet/FFDNet-style noise-level cue),
        so the network can behave adaptively — light touch on clean patches,
        aggressive denoising on noisy ones. Costs one extra conv, negligible
        at inference time.

        refine_blk_num: number of NAFBlocks run at FULL output resolution
        after the PixelShuffle upsample, before the final 1-channel conv.
        0 reproduces the original single-conv upsample head. These blocks
        are the main lever for detail that looks soft/smoothed after the
        2x step — they get a second look at the image at its true output
        resolution instead of only ever seeing it downsampled.
        """
        super().__init__()
        self.aux_noise_map = aux_noise_map
        eff_in_ch = in_ch + (1 if aux_noise_map else 0)

        self.intro = nn.Conv2d(eff_in_ch, width, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.ending = nn.Conv2d(chan, width, 3, padding=1, bias=True)

        # 2x super-resolution head: conv to width*4 channels -> PixelShuffle(2)
        # -> width channels at FULL output resolution -> optional refinement
        # NAFBlocks -> conv to out_ch.
        self.up_shuffle = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
        )
        self.refine = nn.Sequential(*[NAFBlock(width) for _ in range(refine_blk_num)])
        self.up_out = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)

        self.padder_size = 2 ** len(self.encoders)

    def _check_pad(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

    @staticmethod
    def _local_variance(x, k=7):
        pad = k // 2
        mu = F.avg_pool2d(F.pad(x, [pad] * 4, mode="reflect"), k, stride=1)
        mu_sq = F.avg_pool2d(F.pad(x * x, [pad] * 4, mode="reflect"), k, stride=1)
        return (mu_sq - mu * mu).clamp(min=0)

    def forward(self, x):
        """x: (B, 1, H, W) in [0, 1]. Returns (B, 1, 2H, 2W) in [0, 1]."""
        _, _, H, W = x.shape

        inp = x
        if self.aux_noise_map:
            noise_map = self._local_variance(x)
            inp = torch.cat([x, noise_map], dim=1)

        inp = self._check_pad(inp)

        feat = self.intro(inp)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        feat = self.ending(feat)
        feat = self.up_shuffle(feat)
        feat = self.refine(feat)
        correction = self.up_out(feat)
        correction = correction[:, :, : H * 2, : W * 2]

        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        out = base + correction
        return out.clamp(0.0, 1.0)


def parse_blocks(s):
    """'2,2,4' -> (2,2,4). Used for enc_blocks/dec_blocks CLI args."""
    return tuple(int(x) for x in str(s).split(","))


def build_model(width=32, enc_blocks="2,2,4", middle_blocks=6, dec_blocks="2,2,2",
                 refine_blocks=2, aux_noise_map=True, ckpt_args=None):
    """Construct a NAFNetSR2x, preferring architecture settings saved inside
    a loaded checkpoint's 'args' dict (if provided) over the passed-in
    defaults/CLI values. This is what makes --resume, infer.py, and
    visualize_val.py safe to run WITHOUT re-specifying --width/--enc_blocks/
    etc. exactly as they were during training — mismatched flags used to
    cause a state_dict shape-mismatch crash; now the checkpoint wins.
    """
    if ckpt_args:
        width = ckpt_args.get("width", width)
        enc_blocks = ckpt_args.get("enc_blocks", enc_blocks)
        middle_blocks = ckpt_args.get("middle_blocks", middle_blocks)
        dec_blocks = ckpt_args.get("dec_blocks", dec_blocks)
        refine_blocks = ckpt_args.get("refine_blocks", refine_blocks)

    return NAFNetSR2x(
        width=width,
        enc_blk_nums=parse_blocks(enc_blocks),
        middle_blk_num=middle_blocks,
        dec_blk_nums=parse_blocks(dec_blocks),
        refine_blk_num=refine_blocks,
        aux_noise_map=aux_noise_map,
    )


if __name__ == "__main__":
    configs = {
        "default (width=32, refine=2)": dict(width=32, refine_blk_num=2),
        "no refinement (refine=0, old behavior)": dict(width=32, refine_blk_num=0),
        "bigger (width=48, refine=4)": dict(width=48, refine_blk_num=4),
    }
    for name, kwargs in configs.items():
        m = NAFNetSR2x(**kwargs)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"\n{name}: {n_params/1e6:.2f}M params")
        with torch.no_grad():
            for size in (128, 256):
                x = torch.randn(1, 1, size, size).clamp(0, 1)
                y = m(x)
                print(f"  in {tuple(x.shape)} -> out {tuple(y.shape)}")
