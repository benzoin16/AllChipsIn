"""
Diagnostic tool: pick N samples from the held-out validation split, run the
model, and save a side-by-side grid (bicubic-upsampled input | prediction |
ground truth | error map) as PNGs. Look at the error map specifically:

  - Fine speckled/grainy error, spread evenly    -> leftover noise not fully
                                                     suppressed. Try more
                                                     capacity (--width) or
                                                     more training.
  - Error concentrated on edges/fine detail       -> the model is smoothing
                                                     over detail during the
                                                     2x upsample step rather
                                                     than reconstructing it.
                                                     Try --fft_weight higher,
                                                     or more capacity.
  - Error concentrated on specific structure types
    / one image looks much worse than others      -> distribution mismatch,
                                                     likely --synth_prob
                                                     too high/low, or that
                                                     structure type is
                                                     underrepresented.
  - Blocky/tiled artifacts                        -> padding or PixelShuffle
                                                     issue in the model
                                                     itself, worth flagging
                                                     back for a code look.

Usage:
  python visualize_val.py --ckpt checkpoints/nafnetsr2x_best.pt \
                           --data_root ./data --num_samples 6
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import KLARestorationDataset, split_train_val
from model import build_model


def to_display(t):
    """(1,H,W) tensor -> HxW numpy, clipped to [0,1] for imshow (only for
    display purposes — the underlying error map below uses unclipped
    values so out-of-range errors aren't hidden)."""
    return t.squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./val_diagnostics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt_args=ckpt.get("args")).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"loaded {args.ckpt}  (trained to epoch {ckpt.get('epoch', '?')}, "
          f"best_ssim {ckpt.get('best_ssim', '?')})")

    train_root = os.path.join(args.data_root, "train")
    _, val_files = split_train_val(train_root, val_frac=args.val_frac, seed=args.seed)
    val_ds = KLARestorationDataset(train_root, split="val", filenames=val_files)

    n = min(args.num_samples, len(val_ds))
    # spread picks across the val set rather than just the first N, so you
    # see variety rather than whatever happens to sort first
    idxs = np.linspace(0, len(val_ds) - 1, n, dtype=int)

    for rank, idx in enumerate(idxs):
        lr, gt = val_ds[idx]
        fname = os.path.basename(val_ds.lr_paths[idx])

        lr_b, gt_b = lr.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(lr_b).clamp(0.0, 1.0)

        bicubic = F.interpolate(lr_b.clamp(0, 1), scale_factor=2, mode="bicubic", align_corners=False)

        pred_np = to_display(pred[0])
        gt_np = to_display(gt_b[0])
        bicubic_np = to_display(bicubic[0])
        error_np = np.abs(pred_np - gt_np)  # both already clamped to [0,1]

        psnr_val = 10 * np.log10(1.0 / (np.mean((pred_np - gt_np) ** 2) + 1e-10))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        titles = [
            f"Bicubic input (no model)",
            f"Prediction",
            f"Ground truth",
            f"|error|  (pSNR {psnr_val:.2f} dB)",
        ]
        images = [bicubic_np, pred_np, gt_np, error_np]
        cmaps = ["gray", "gray", "gray", "inferno"]
        for ax, img, title, cmap in zip(axes, images, titles, cmaps):
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap == "gray" else error_np.max())
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            if cmap == "inferno":
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{fname}  (LR {tuple(lr.shape[-2:])} -> {tuple(gt.shape[-2:])})", fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"sample_{rank:02d}_{fname.replace('.npy', '')}.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"saved {out_path}  (pSNR {psnr_val:.2f} dB)")

    print(f"\nall {n} diagnostic images saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
