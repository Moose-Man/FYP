#!/usr/bin/env python3
"""
Generate ONE big comparison figure:

    filename ─ H&E (orig) ─ ver-1 ─ ver-2 ─ … ─ IHC (GT)

… and print PSNR & SSIM for each generated version.

Dependencies:
    pip install pillow matplotlib tqdm numpy scikit-image
"""

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim

# -------------------------------------------------------------------------- #
#                                ### CONFIG ###                              #
# -------------------------------------------------------------------------- #
HE_DIR   = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train")
GT_DIR   = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train")
GEN_DIRS = [
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_1\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_2\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_3\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_4\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_5\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_6\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_7\epoch_50"),
]
OUT_DIR      = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\data_visualization\train_dataset")
EXTENSIONS   = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIGSIZE      = (20, 4)      # inches per row
DPI          = 150

SELECTED_FILES_TEST = [
    "00082_test_0.png",
    "00628_test_0.png",
    "00995_test_0.png",
    "00010_test_1+.png",
    "00068_test_1+.png",
    "00108_test_1+.png",
    "00156_test_2+.png",
    "00190_test_2+.png",
    "00337_test_2+.png",
    "00460_test_3+.png",
    "00531_test_3+.png",
    "00686_test_3+.png"
]

SELECTED_FILES_TRAIN = [
    "03918_train_0.png",
    "03676_train_0.png",
    "02258_train_0.png",
    "01347_train_1+.png",
    "01087_train_1+.png",
    "00811_train_1+.png",
    "00631_train_2+.png",
    "00575_train_2+.png",
    "00469_train_2+.png",
    "03837_train_3+.png",
    "03645_train_3+.png",
    "03387_train_3+.png"
]
# -------------------------------------------------------------------------- #

LABELS  = ["H&E (orig)"] + [f"ver‑{i+1} (gen)" for i in range(len(GEN_DIRS))] + ["IHC (GT)"]
FOLDERS = [HE_DIR] + GEN_DIRS + [GT_DIR]


# -------------------------------------------------------------------------- #
#                              Helper functions                              #
# -------------------------------------------------------------------------- #
def load_images(fname: str, roots: List[Path]):
    """Return list of PIL Images in the folder order defined by *roots*."""
    imgs = []
    for root in roots:
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"{fname!r} missing in {root}")
        imgs.append(Image.open(path).convert("RGB"))
    return imgs


def compute_metrics(filenames: List[str]):
    """Compute & print PSNR / SSIM for every generated version."""
    print("\n=== Image quality metrics (vs. ground‑truth IHC) ===")
    header = ["file"] + [f"ver‑{i+1}" for i in range(len(GEN_DIRS))]
    print("{:<20}".format(header[0]), end="")
    for h in header[1:]:
        print(f"{h:^26}", end="")
    print("\n" + "-" * (26 * len(GEN_DIRS) + 20))

    for fname in filenames:
        # Load GT once
        gt_arr = np.array(Image.open(GT_DIR / fname).convert("RGB"))
        print(f"{fname:<20}", end="")

        for gdir in GEN_DIRS:
            gen_arr = np.array(Image.open(gdir / fname).convert("RGB"))
            p = psnr(gt_arr, gen_arr, data_range=255)
            s = ssim(gt_arr, gen_arr, channel_axis=-1, data_range=255)
            print(f"PSNR:{p:5.1f}  SSIM:{s:4.3f}", end="  ")
        print()  # newline per image


def make_stacked_figure(filenames: List[str]):
    """Create single stacked figure & save to OUT_DIR/all_strips.png."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows        = len(filenames)
    n_img_cols    = len(FOLDERS)
    n_total_cols  = n_img_cols + 1   # extra col for filename labels

    fig_width  = FIGSIZE[0]
    fig_height = FIGSIZE[1] * n_rows * 0.5
    fig, axes  = plt.subplots(
        n_rows, n_total_cols, figsize=(fig_width, fig_height), squeeze=False
    )

    # Column headers (skip filename column)
    for j, label in enumerate(LABELS, start=1):
        axes[0, j].set_title(label, fontsize=12, pad=12)

    # Populate rows
    for row, fname in enumerate(tqdm(filenames, desc="Building figure", unit="img")):
        imgs = load_images(fname, FOLDERS)

        # Filename cell
        axes[row, 0].axis("off")
        axes[row, 0].text(0.5, 0.5, fname, ha="center", va="center", fontsize=9)

        # Image cells
        for col, img in enumerate(imgs, start=1):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / "all_strips.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------------- #
#                                  main()                                    #
# -------------------------------------------------------------------------- #
def main():
    files = SELECTED_FILES_TRAIN
    if not files:
        raise RuntimeError("SELECTED_FILES list is empty!")

    # Verify every requested file exists in every folder
    for fname in files:
        for root in FOLDERS:
            if not (root / fname).exists():
                raise FileNotFoundError(f"{fname!r} not found in {root}")

    # Metrics first (prints to terminal)
    compute_metrics(files)

    # Build the consolidated visualisation
    print("\nCreating stacked figure …")
    out_path = make_stacked_figure(files)
    print(f"\n✅  Finished. Figure saved → {out_path}")


if __name__ == "__main__":
    main()
