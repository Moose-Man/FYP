#!/usr/bin/env python
"""
Compute average PSNR and SSIM between two directories of images.

Usage example
-------------
python evaluate_psnr_ssim.py \
    --gt_dir  "C:/data/BCI_dataset/IHC_resized_nocrop/test" \
    --pred_dir "C:/runs/he2ihc_bicycle_baseline/test_images"

The script looks for files with the same *basename* in both folders
(e.g. 00012.png in gt_dir and 00012.png in pred_dir).
Supported extensions: .png .jpg .jpeg .tif .tiff .bmp
"""

import argparse, os, sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim


def load_rgb(path: Path) -> np.ndarray:
    """Load image as uint8 RGB numpy array (H,W,3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def main(args):
    gt_dir   = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.is_dir() or not pred_dir.is_dir():
        sys.exit("❌ One or both directories do not exist.")

    # Gather all ground-truth filenames
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    gt_files = [p for p in gt_dir.iterdir() if p.suffix.lower() in exts]

    if not gt_files:
        sys.exit("❌ No images found in ground-truth directory.")

    psnr_list, ssim_list = [], []
    missing = 0

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.is_file():
            missing += 1
            if args.verbose:
                print(f"[skip] {gt_path.name} ‒ no match in pred_dir")
            continue

        # load images
        gt   = load_rgb(gt_path)
        out  = load_rgb(pred_path)

        # --- NEW: auto-resize the prediction if dimensions differ --------------
        if gt.shape != out.shape:
            out = np.array(Image.fromarray(out).resize(gt.shape[1::-1], Image.BICUBIC))
        # -----------------------------------------------------------------------

        psnr_val = psnr(gt, out, data_range=255)
        ssim_val = ssim(gt, out, channel_axis=2, data_range=255)


        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        if args.verbose:
            print(f"{gt_path.name:>25} | PSNR {psnr_val:5.2f}  SSIM {ssim_val:6.4f}")

    if not psnr_list:
        sys.exit("❌ No pairs evaluated. Check file names and directories.")

    print("\n==========  Summary  ==========")
    print(f"Images evaluated : {len(psnr_list)}")
    if missing:
        print(f"Missing pairs    : {missing}")
    print(f"Average PSNR     : {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM     : {np.mean(ssim_list):.4f}")
    print("================================")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir",   required=True, help="folder with ground-truth images")
    p.add_argument("--pred_dir", required=True, help="folder with generated images")
    p.add_argument("--verbose",  action="store_true", help="print per-image metrics")
    args = p.parse_args()
    main(args)
