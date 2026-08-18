"""
Restoration-quality metrics matching what the challenge scores on:
SSIM, pSNR, LPIPS. pSNR/SSIM are implemented directly in torch (no extra
deps). LPIPS needs the `lpips` package + a small pretrained weight download
on first use — wrapped so its absence doesn't break validation, it just
gets skipped with a one-time warning.

All functions expect (B, 1, H, W) float32 tensors already clamped to [0, 1]
(that's what model.py's output head guarantees) and matching shapes.
"""
import torch
import torch.nn.functional as F

_LPIPS_MODEL = None
_LPIPS_WARNED = False


def psnr(pred, target, max_val=1.0, eps=1e-10):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))  # (B,)


def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window[None, None, :, :]


def ssim(pred, target, window_size=11, data_range=1.0):
    """Standard single-scale SSIM, single-channel, gaussian window."""
    device = pred.device
    window = _gaussian_window(window_size, device=device)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad)
    mu_t = F.conv2d(target, window, padding=pad)
    mu_p_sq, mu_t_sq, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, window, padding=pad) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, window, padding=pad) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])  # (B,)


def lpips_distance(pred, target):
    """Returns None (not an error) if the lpips package isn't installed —
    caller decides whether to report it as unavailable."""
    global _LPIPS_MODEL, _LPIPS_WARNED
    try:
        import lpips
    except ImportError:
        if not _LPIPS_WARNED:
            print("[metrics] `lpips` package not installed — skipping LPIPS "
                  "(pip install lpips to enable it).")
            _LPIPS_WARNED = True
        return None

    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(pred.device)
        _LPIPS_MODEL.eval()
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad_(False)

    # lpips expects 3-channel input in [-1, 1]
    pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
    tgt3 = target.repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        d = _LPIPS_MODEL(pred3, tgt3)
    return d.view(-1)  # (B,)


@torch.no_grad()
def compute_all(pred, target):
    """Returns a dict of {metric_name: (B,) tensor}. LPIPS key omitted if
    the package isn't available."""
    out = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
    lp = lpips_distance(pred, target)
    if lp is not None:
        out["lpips"] = lp
    return out
