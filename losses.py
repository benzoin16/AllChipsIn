import torch.nn.functional as F
import torch

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def fft_loss(pred, target):
    pred_f = torch.fft.rfft2(pred, norm="ortho")
    target_f = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))

def gaussian_blur(x, sigma=1.5, ksize=5):
    device, dtype = x.device, x.dtype
    coords = torch.arange(ksize, dtype=dtype, device=device) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] @ g[None, :])[None, None]
    pad = ksize // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), kernel)

def hf_residual_loss(pred, target):
    pred_hf = pred - gaussian_blur(pred)
    target_hf = target - gaussian_blur(target)
    return F.l1_loss(pred_hf, target_hf)

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [0.0, 0.0, 0.0],
     [1.0, 2.0, 1.0]]
)

def edge_loss(pred, target):
    device, dtype = pred.device, pred.dtype
    kx = _SOBEL_X.to(device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=device, dtype=dtype).view(1, 1, 3, 3)

    pred_gx = F.conv2d(pred, kx, padding=1)
    pred_gy = F.conv2d(pred, ky, padding=1)
    target_gx = F.conv2d(target, kx, padding=1)
    target_gy = F.conv2d(target, ky, padding=1)

    pred_edges = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
    target_edges = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

    return F.l1_loss(pred_edges, target_edges)

_LPIPS_MODEL = None

def get_lpips_loss(pred, target):
    global _LPIPS_MODEL

    if _LPIPS_MODEL is None:

        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(pred.device)
        _LPIPS_MODEL.eval()

        for p in _LPIPS_MODEL.parameters():
            p.requires_grad_(False)

    pred3 = pred.repeat(1, 3, 1, 1) * 2 - 1
    target3 = target.repeat(1, 3, 1, 1) * 2 - 1

    return _LPIPS_MODEL(pred3, target3).mean()