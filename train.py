import argparse
import os
import time

import numpy as np
from losses import *
import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import KLARestorationDataset, split_train_val
from model import build_model
import metrics

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[k] = v.detach().clone()

    def store(self, model):
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("EMA.restore() called without EMA.store().")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

@torch.no_grad()
def run_validation(model, val_dl, device):
    model.eval()

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    has_lpips = False
    n = 0

    for lr_img, gt_img in val_dl:
        lr_img = lr_img.to(device, non_blocking=True)
        gt_img = gt_img.to(device, non_blocking=True).clamp(0.0, 1.0)

        pred = pred.float().clamp(0.0, 1.0)

        m = metrics.compute_all(pred, gt_img)
        batch_size = pred.shape[0]

        totals["psnr"] += m["psnr"].sum().item()
        totals["ssim"] += m["ssim"].sum().item()

        if "lpips" in m:
            has_lpips = True
            totals["lpips"] += m["lpips"].sum().item()

        n += batch_size

    model.train()

    results = {
        "psnr": totals["psnr"] / n,
        "ssim": totals["ssim"] / n,
    }

    if has_lpips:
        results["lpips"] = totals["lpips"] / n

    return results

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr_patch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=7e-4)

    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--enc_blocks", type=str, default="2,2,4")
    ap.add_argument("--middle_blocks", type=int, default=6)
    ap.add_argument("--dec_blocks", type=str, default="2,2,2")
    ap.add_argument("--refine_blocks", type=int, default=2)

    ap.add_argument("--fft_weight", type=float, default=0.05)
    ap.add_argument("--edge_weight", type=float, default=0.1)
    ap.add_argument("--hf_residual_weight", type=float, default=0.1)
    ap.add_argument("--ssim_weight", type=float, default=0.1)
    ap.add_argument("--lpips_weight", type=float, default=0.0)
    ap.add_argument("--mse_weight", type=float, default=0.0)

    ap.add_argument("--synth_prob", type=float, default=0.3)
    ap.add_argument("--cutblur_prob", type=float, default=0.0)
    ap.add_argument("--mixup_prob", type=float, default=0.0)

    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true", default=False)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_root = os.path.join(args.data_root, "train")
    train_files, val_files = split_train_val(
        train_root,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print(f"device: {device}")
    print(f"train pairs: {len(train_files)}  val pairs: {len(val_files)}")

    train_ds = KLARestorationDataset(
        train_root,
        split="train",
        filenames=train_files,
        lr_patch=args.lr_patch,
        synth_prob=args.synth_prob,
        cutblur_prob=args.cutblur_prob,
        mixup_prob=args.mixup_prob,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_ds = KLARestorationDataset(
        train_root,
        split="val",
        filenames=val_files,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, args.num_workers),
    )

    ckpt = None

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=device)
        print(f"[resume] loading {args.resume}")

        saved_arch = ckpt.get("args", {})
        mismatches = {
            k: (saved_arch.get(k), getattr(args, k))
            for k in ("width", "enc_blocks", "middle_blocks", "dec_blocks", "refine_blocks")
            if k in saved_arch and saved_arch.get(k) != getattr(args, k)
        }

        if mismatches:
            print(f"[resume] checkpoint architecture wins: {mismatches}")

    model = build_model(
        width=args.width,
        enc_blocks=args.enc_blocks,
        middle_blocks=args.middle_blocks,
        dec_blocks=args.dec_blocks,
        refine_blocks=args.refine_blocks,
        ckpt_args=ckpt.get("args") if ckpt else None,
    ).to(device).to(memory_format=torch.channels_last)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        print("[resume] model weights restored")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.9),
        weight_decay=1e-4,
    )

    if ckpt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
        print("[resume] optimizer state restored")
    else:
        print("[resume] fresh AdamW optimizer")

    for group in opt.param_groups:
        group["lr"] = args.lr

    start_epoch = ckpt.get("epoch", -1) + 1 if ckpt else 0
    best_ssim = ckpt.get("best_ssim", -1.0) if ckpt else -1.0

    print(
        f"[resume] epoch={start_epoch}  "
        f"lr={opt.param_groups[0]['lr']:.2e}"
    )

    ema = EMA(model, decay=args.ema_decay)

    if ckpt is not None and "ema" in ckpt:
        ema.shadow = {
            k: v.detach().clone()
            for k, v in ckpt["ema"].items()
        }
        print("[resume] EMA state restored")
    else:
        print("[resume] fresh EMA initialized")

    remaining_epochs = max(1, args.epochs - start_epoch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=remaining_epochs * len(train_dl),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0

        for lr_img, gt_img in train_dl:
            lr_img = lr_img.to(
                device,
                non_blocking=True,
            ).to(memory_format=torch.channels_last)

            gt_img = gt_img.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                pred = model(lr_img)

                loss = charbonnier_loss(pred, gt_img)

                if args.fft_weight > 0:
                    loss += args.fft_weight * fft_loss(pred, gt_img)

                if args.edge_weight > 0:
                    loss += args.edge_weight * edge_loss(pred, gt_img)

                if args.ssim_weight > 0:
                    loss += args.ssim_weight * (
                        1.0 - metrics.ssim(pred, gt_img).mean()
                    )

                if args.lpips_weight > 0:
                    loss += args.lpips_weight * get_lpips_loss(pred, gt_img)

                if args.mse_weight > 0:
                    loss += args.mse_weight * F.mse_loss(pred, gt_img)

                if args.hf_residual_weight > 0:
                    loss += args.hf_residual_weight * hf_residual_loss(pred, gt_img)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(opt)
            scaler.update()

            ema.update(model)
            sched.step()

            running += loss.item()

        avg_loss = running / max(1, len(train_dl))
        current_lr = opt.param_groups[0]["lr"]

        print(
            f"epoch {epoch:04d}  "
            f"loss {avg_loss:.10f}  "
            f"lr {current_lr:.3e}  "
            f"time {time.time() - t0:.1f}s"
        )

        run_val = ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1)

        if run_val:
            t_val = time.time()

            ema.store(model)
            ema.apply_to(model)

            val_metrics = run_validation(
                model,
                val_dl,
                device,
               
            )

            ema.restore(model)

            msg = (
                f"psnr {val_metrics['psnr']:.3f}  "
                f"ssim {val_metrics['ssim']:.4f}"
            )
            
            if "lpips" in val_metrics:
                msg += f"  lpips {val_metrics['lpips']:.4f}"

            msg += f"  ({time.time() - t_val:.1f}s)"
            print(msg)

            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]

                best_path = os.path.join(
                    args.ckpt_dir,
                    "nafnetsr2x_best.pt",
                )

                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.shadow,
                        "opt": opt.state_dict(),
                        "epoch": epoch,
                        "best_ssim": best_ssim,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    best_path,
                )

                print(
                    f"  new best ssim {best_ssim:.4f} "
                    f"-> saved {best_path}"
                )

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            path = os.path.join(
                args.ckpt_dir,
                f"nafnetsr2x_e{epoch:04d}.pt",
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "best_ssim": best_ssim,
                    "args": vars(args),
                },
                path,
            )

            print(f"  saved {path}")

if __name__ == "__main__":
    main()