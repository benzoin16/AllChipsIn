import argparse
import glob
import os
import time

import numpy as np
import torch

from model import build_model


def load_npy(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    return np.asarray(arr, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Run NAFNetSR2x inference on a directory of .npy LR images.")
    ap.add_argument("input_dir", required=True)
    ap.add_argument("output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    # Use BooleanOptionalAction to allow both --fp16 and --no-fp16 flags
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--compile", action="store_true", help="enable torch.compile()")
    args = ap.parse_args()

    model_path = glob.glob("models/*.pt")
    args.ckpt = model_path[0]

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[info] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model(ckpt_args=ckpt.get("args")).to(device)

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=True)
        print("[info] using EMA weights")
    else:
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)

    if args.fp16 and device.type == "cuda":
        model.half()
        print("[info] Model converted to FP16")
    model.eval()

    if args.compile:
        print("[info] compiling model with torch.compile()...")
        # dynamic=True prevents recompilation overhead when processing different image shapes
        model = torch.compile(model, dynamic=True)

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    n = len(paths)

    if n == 0:
        print(f"\n[error] No .npy files found in {args.input_dir}")
        print("Please verify the --input_dir path.")
        return

    # Memory Optimization: Only read shapes during the initial scan using memory mapping.
    # We store the paths grouped by shape, but we DO NOT hold the array data in RAM.
    print(f"[info] Scanning {n} files for shape grouping...")
    by_shape = {}
    for path in paths:
        try:
            # mmap_mode='r' reads the header to get the shape without loading the full array
            shape = np.load(path, mmap_mode='r', allow_pickle=True).shape
        except Exception:
            # Fallback for complex object arrays if mmap fails
            shape = load_npy(path).shape
        by_shape.setdefault(shape, []).append(path)

    t_setup_done = time.time()
    print("[info] Starting inference...")

    with torch.inference_mode():
        for shape, path_group in by_shape.items():
            for i in range(0, len(path_group), args.batch_size):
                chunk_paths = path_group[i:i + args.batch_size]
                
                # Lazy load the arrays only when they are needed for the current batch
                arrays = [load_npy(p) for p in chunk_paths]
                batch_np = np.stack(arrays, axis=0)[:, None, :, :]
                
                # Use .pin_memory() before .to(non_blocking=True) for true async transfers
                batch = torch.from_numpy(batch_np).pin_memory().to(device, non_blocking=True)
                
                if args.fp16 and device.type == "cuda":
                    batch = batch.half()

                out = model(batch).float().cpu().numpy()

                out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                out = np.clip(out, 0.0, 1.0)

                for j, path in enumerate(chunk_paths):
                    out_path = os.path.join(args.output_dir, os.path.basename(path))
                    # out[j, 0] gives a grayscale (H, W) array
                    np.save(out_path, out[j, 0].astype(np.float32))

    t_end = time.time()

    # Print summary statistics
    print("\n--- Inference Summary ---")
    print(f"Images Processed: {n}")
    print(f"FP16 Enabled:     {args.fp16 and device.type == 'cuda'}")
    print(f"Startup Time:     {t_setup_done - t_start:.3f}s")
    print(f"Inference+IO:     {t_end - t_setup_done:.3f}s")
    print(f"Total Time:       {t_end - t_start:.3f}s  ({(t_end - t_start) / n * 1000:.2f} ms/img)")


if __name__ == "__main__":
    main()