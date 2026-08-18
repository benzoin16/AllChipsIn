"""
Paired GT / NoisyLR dataset loader for the actual release format:

  root/
    GT/       000000.npy ... 003199.npy   float32, ~[0, 1], clean, HxW
    NoisyLR/  000000.npy ... 003199.npy   float32, CAN exceed [0, 1]
                                          (speckle pushes values above/below
                                          the ground-truth range), H/2 x W/2

Confirmed from inspect_npy.py: NoisyLR/000000.npy is (128,128) float32 with
min=0.0010, max=1.5406 — i.e. NOT a 0-255 image, already normalized, with
genuine out-of-range excursions baked in. Because of this:

  - We do NOT go through PIL/uint8 anywhere (that would silently clip the
    exact signal we're supposed to learn to suppress).
  - All resizing/augmentation happens directly on the float32 arrays via
    numpy / torch, preserving out-of-range values until they hit the model,
    whose output head clamps to [0, 1] (see model.py) — the input is allowed
    to be out of range, the *target* and the *prediction* are not.
  - The 512x512-GT/256-LR pairs and 256-GT/128-LR pairs are just two sizes
    of the same x2 problem; nothing here assumes a fixed resolution.

Test set: same NoisyLR/ layout, no GT/ (that's what you're predicting).
Use split="test" to load NoisyLR-only (returns lr, filename).

If your actual layout differs (e.g. GT lives elsewhere, or filenames don't
line up 1:1 between the two folders), the two path-building lines flagged
below with "ADAPT HERE" are the only things that need to change.
"""
import glob
import os
import random

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()
    return np.asarray(arr, dtype=np.float32)


def _random_crop_pair(gt, lr, lr_patch=96):
    """gt is exactly 2x lr's size in both dims."""
    lh, lw = lr.shape
    if lh < lr_patch or lw < lr_patch:
        pad_h, pad_w = max(0, lr_patch - lh), max(0, lr_patch - lw)
        lr = np.pad(lr, ((0, pad_h), (0, pad_w)), mode="reflect")
        gt = np.pad(gt, ((0, pad_h * 2), (0, pad_w * 2)), mode="reflect")
        lh, lw = lr.shape
    y = random.randint(0, lh - lr_patch)
    x = random.randint(0, lw - lr_patch)
    lr_c = lr[y:y + lr_patch, x:x + lr_patch]
    gt_c = gt[y * 2:(y + lr_patch) * 2, x * 2:(x + lr_patch) * 2]
    return gt_c, lr_c

def _augment(gt, lr):
    if random.random() < 0.5:
        gt, lr = gt[:, ::-1].copy(), lr[:, ::-1].copy()
    if random.random() < 0.5:
        gt, lr = gt[::-1, :].copy(), lr[::-1, :].copy()
    k = random.randint(0, 3)
    if k:
        gt, lr = np.rot90(gt, k).copy(), np.rot90(lr, k).copy()
    return gt, lr


def _torch_resize_half(img_f32, mode):
    """Downsample a HxW float32 array by 2x using torch, no PIL/cv2 involved
    so out-of-range values pass through untouched (mode in {'bicubic',
    'bilinear', 'area'})."""
    t = torch.from_numpy(img_f32)[None, None, :, :]
    h, w = t.shape[-2:]
    kwargs = {} if mode == "area" else {"align_corners": False}
    out = F.interpolate(t, size=(h // 2, w // 2), mode=mode, **kwargs)
    return out[0, 0].numpy()


def _gaussian_blur(img_f32, sigma):
    t = torch.from_numpy(np.asarray(img_f32, dtype=np.float32))[None, None, :, :]
    ksz = int(2 * round(sigma * 2) + 1)
    coords = torch.arange(ksz, dtype=torch.float32) - ksz // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(torch.float32)
    kernel = (g[:, None] @ g[None, :])[None, None, :, :]
    pad = ksz // 2
    out = F.conv2d(F.pad(t, [pad] * 4, mode="reflect"), kernel)
    return out[0, 0].numpy()


def _synth_degrade(gt_f32):
    h, w = gt_f32.shape
    gt_f32 = gt_f32[: h - h % 2, : w - w % 2]
    img = gt_f32.copy()

    def do_blur(im):
        sigma = random.uniform(0.6, 1.6)
        return _gaussian_blur(im, sigma)

    def do_downsample(im):
        kernel = random.choice(["bicubic", "bilinear", "area"])
        return _torch_resize_half(im, kernel)

    def do_speckle(im):
        var = random.uniform(0.005, 0.04)
        return im + im * np.random.randn(*im.shape).astype(np.float32) * np.sqrt(var)

    def do_gaussian(im):
        sigma = random.uniform(0.01, 0.06)
        return im + np.random.randn(*im.shape).astype(np.float32) * sigma

    steps = [("downsample", do_downsample)] 
    if random.random() < 0.7:
        steps.append(("blur", do_blur))
    if random.random() < 0.6:
        steps.append(("speckle", do_speckle))
    if random.random() < 0.6:
        steps.append(("gaussian", do_gaussian))

    random.shuffle(steps)

    for _, fn in steps:
        img = fn(img)

    return gt_f32.astype(np.float32), img.astype(np.float32)


def _apply_cutblur(gt, lr, prob=0.5):
    """gt: numpy float32 HxW, lr: numpy float32 (H/2)x(W/2).

    CutBlur (Yoo et al. 2020) only ever mixes content into the model's
    INPUT — the ground-truth label must stay pristine the whole time, since
    the entire point is teaching the network to recognize "this region is
    already clean, don't touch it" vs "this region needs restoration",
    which only works if what it's scored against during training is always
    the true clean answer. gt is therefore never modified here — only lr
    (the input) gets a patch of real GT content pasted in, then gets
    downsampled back to LR resolution so the returned lr stays at the
    correct input resolution for the model."""
    if random.random() > prob:
        return gt, lr

    h, w = gt.shape
    t_lr = torch.from_numpy(lr)[None, None, :, :]
    lr_up = F.interpolate(t_lr, size=(h, w), mode="bicubic", align_corners=False)[0, 0].numpy()

    bh = random.randint(int(h * 0.2), int(h * 0.8))
    bw = random.randint(int(w * 0.2), int(w * 0.8))
    by = random.randint(0, h - bh)
    bx = random.randint(0, w - bw)

    mask = np.zeros_like(gt)
    mask[by:by + bh, bx:bx + bw] = 1.0

    # paste real clean GT content into the upsampled-LR input, then
    # downsample back to LR resolution — gt itself is NEVER touched
    lr_up_mixed = mask * gt + (1.0 - mask) * lr_up
    lr = _torch_resize_half(lr_up_mixed, mode="bicubic")

    return gt.astype(np.float32), lr.astype(np.float32)


def _apply_mixup(gt1, lr1, gt2, lr2):
    alpha = random.uniform(0.1, 0.9)
    gt = alpha * gt1 + (1.0 - alpha) * gt2
    lr = alpha * lr1 + (1.0 - alpha) * lr2
    return gt.astype(np.float32), lr.astype(np.float32)


def split_train_val(root, val_frac=0.1, seed=42):
    """Splits the filenames under root/NoisyLR/ into (train_filenames,
    val_filenames), holding out val_frac of them deterministically (fixed
    seed -> same split every run, so results are comparable across training
    runs and resumes).

    root should be the folder containing GT/ and NoisyLR/ (e.g.
    <data_root>/train), NOT the top-level data_root — the test set has no
    GT/ to validate against, so validation is always carved out of train.
    """
    all_paths = sorted(glob.glob(os.path.join(root, "NoisyLR", "*.npy")))
    assert all_paths, f"no .npy files found under {root}/NoisyLR"
    filenames = [os.path.basename(p) for p in all_paths]

    rng = random.Random(seed)
    shuffled = filenames[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_frac))
    val_files = sorted(shuffled[:n_val])
    train_files = sorted(shuffled[n_val:])
    return train_files, val_files


class KLARestorationDataset(Dataset):
    def __init__(self, root, split="train", lr_patch=96, synth_prob=0.5, filenames=None, cutblur_prob=0.0, mixup_prob=0.0):
        """
        root: directory containing GT/ and NoisyLR/ subfolders.
        split: "train" -> returns (lr, gt) tensors, with cropping/aug/synth.
               "val"   -> returns (lr, gt) tensors, full images, no aug/synth.
               "test"  -> returns (lr, filename); no GT/ required.
        filenames: optional explicit list of basenames (e.g. ["000003.npy", ...])
               to restrict this dataset to — this is how train/val splits
               share one root without overlapping. If None, uses every file
               found under NoisyLR/.
        """
        self.root = root
        self.split = split
        self.lr_patch = lr_patch
        self.synth_prob = synth_prob if split == "train" else 0.0
        self.cutblur_prob = cutblur_prob if split == "train" else 0.0
        self.mixup_prob = mixup_prob if split == "train" else 0.0

        if filenames is not None:
            # ADAPT HERE if your NoisyLR folder is named differently.
            self.lr_paths = [os.path.join(root, "NoisyLR", f) for f in filenames]
            missing = [p for p in self.lr_paths if not os.path.exists(p)]
            assert not missing, f"{len(missing)} NoisyLR files missing, e.g. {missing[:3]}"
        else:
            # ADAPT HERE if your NoisyLR folder is named differently.
            self.lr_paths = sorted(glob.glob(os.path.join(root, "NoisyLR", "*.npy")))
            assert self.lr_paths, f"no .npy files found under {root}/NoisyLR"

        if split in ("train", "val"):
            # ADAPT HERE if your GT folder is named differently or filenames
            # don't match 1:1 with NoisyLR (e.g. different index scheme).
            self.gt_paths = [os.path.join(root, "GT", os.path.basename(p)) for p in self.lr_paths]
            missing = [p for p in self.gt_paths if not os.path.exists(p)]
            assert not missing, f"{len(missing)} GT files missing, e.g. {missing[:3]}"

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        if self.split == "test":
            lr = _load(self.lr_paths[idx])
            return torch.from_numpy(lr)[None, ...], os.path.basename(self.lr_paths[idx])

        gt_f = _load(self.gt_paths[idx])
        if self.split == "train" and random.random() < self.synth_prob:
            gt_f, lr_f = _synth_degrade(gt_f)
        else:
            lr_f = _load(self.lr_paths[idx])

        if self.split == "train":
            gt_f, lr_f = _random_crop_pair(gt_f, lr_f, self.lr_patch)
            
            # Apply MixUp
            if self.mixup_prob > 0.0 and random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self.lr_paths) - 1)
                gt_f2 = _load(self.gt_paths[idx2])
                if random.random() < self.synth_prob:
                    gt_f2, lr_f2 = _synth_degrade(gt_f2)
                else:
                    lr_f2 = _load(self.lr_paths[idx2])
                gt_f2, lr_f2 = _random_crop_pair(gt_f2, lr_f2, self.lr_patch)
                gt_f, lr_f = _apply_mixup(gt_f, lr_f, gt_f2, lr_f2)

            # Apply CutBlur
            if self.cutblur_prob > 0.0:
                gt_f, lr_f = _apply_cutblur(gt_f, lr_f, self.cutblur_prob)

            gt_f, lr_f = _augment(gt_f, lr_f)
        # val: use full images as-is (no crop) so metrics reflect real eval

        gt_t = torch.from_numpy(np.ascontiguousarray(gt_f)).float()[None, ...]
        lr_t = torch.from_numpy(np.ascontiguousarray(lr_f)).float()[None, ...]
        return lr_t, gt_t
