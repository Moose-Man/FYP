"""
Generate comparison figures and metrics for HER2 test images:
 - Samples n files per HER2 level (0, 1+, 2+, 3+)
 - Computes PSNR & SSIM vs. ground-truth IHC
 - Builds a grouped strip figure by HER2 level
 - Builds a tight side-by-side figure across all versions
 - Prints metrics in a single‐column format:
     HER2 level X
     filename - ver-Y - PSNR - SSIM
     …
"""
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim

# -------------------------------------------------------------------------- #
#                                CONFIGURATION                               #
# -------------------------------------------------------------------------- #
HE_DIR   = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\test")
GT_DIR   = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\test")
GEN_DIRS = [
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_1\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_2\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_3\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_4\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_5\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_6\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_7\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_8\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_9\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_10\epoch_50"),
    Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\test_dataset\final assortment\ver_11\epoch_50"),
]
OUT_DIR      = Path(r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\data_visualization\test_dataset")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIGSIZE    = (20, 4)     # width, height per unit row in inches
DPI        = 150

HER2_LEVELS = ["0", "1+", "2+", "3+"]

# -------------------------------------------------------------------------- #
#                              ARGUMENT PARSING                              #
# -------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Sample by HER2 level, compute metrics, and build comparison figures"
    )
    p.add_argument(
        "-n", "--num-samples", type=int, default=5,
        help="Number of random files to sample per HER2 level"
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    return p.parse_args()

# -------------------------------------------------------------------------- #
#                          HER2 LEVEL EXTRACTION                            #
# -------------------------------------------------------------------------- #
def extract_level(filename: str) -> str:
    """
    Extract HER2 level (0, 1+, 2+, 3+) from filename tokens.
    """
    tokens = re.split(r'[^0-9\+]+', filename)
    for t in tokens:
        if t in HER2_LEVELS:
            return t
    for lvl in HER2_LEVELS:
        if lvl in filename:
            return lvl
    raise ValueError(f"Cannot extract HER2 level from filename: {filename}")

# -------------------------------------------------------------------------- #
#                           SAMPLING BY HER2 LEVEL                           #
# -------------------------------------------------------------------------- #
def sample_by_level(gen_dirs: List[Path], n: int) -> Dict[str, List[str]]:
    he_files = {p.name for p in HE_DIR.iterdir() if p.suffix.lower() in EXTENSIONS}
    gt_files = {p.name for p in GT_DIR.iterdir() if p.suffix.lower() in EXTENSIONS}
    common   = he_files & gt_files
    for g in gen_dirs:
        common &= {p.name for p in g.iterdir() if p.suffix.lower() in EXTENSIONS}
    if not common:
        raise RuntimeError("No overlapping files across HE, GT, and generated dirs!")

    by_level: Dict[str, List[str]] = {lvl: [] for lvl in HER2_LEVELS}
    for fname in sorted(common):
        lvl = extract_level(fname)
        by_level[lvl].append(fname)

    sampled: Dict[str, List[str]] = {}
    for lvl, files in by_level.items():
        if not files:
            raise RuntimeError(f"No files found for HER2 level '{lvl}'")
        sampled[lvl] = files if len(files) <= n else random.sample(files, n)
    return sampled

# -------------------------------------------------------------------------- #
#                         VERSION LABEL GENERATION                           #
# -------------------------------------------------------------------------- #
def make_version_labels(gen_dirs: List[Path]) -> List[str]:
    labels = []
    for g in gen_dirs:
        parent = g.parent.name   # e.g. "ver_3"
        parts  = parent.split("_", 1)
        ver    = parts[1] if len(parts) == 2 else parent
        labels.append(f"ver-{ver} (gen)")
    return labels

# -------------------------------------------------------------------------- #
#                              METRIC COMPUTATION                            #
# -------------------------------------------------------------------------- #
def compute_metrics(samples_by_level: Dict[str, List[str]], gen_dirs: List[Path]):
    for lvl, flist in samples_by_level.items():
        print(f"\nHER2 level {lvl}")
        for fname in flist:
            # load once per file
            gt_arr = np.array(Image.open(GT_DIR / fname).convert("RGB"))
            for gdir in gen_dirs:
                arr = np.array(Image.open(gdir / fname).convert("RGB"))
                p   = psnr(gt_arr, arr, data_range=255)
                s   = ssim(gt_arr, arr, channel_axis=-1, data_range=255)
                ver = gdir.parent.name.split("_", 1)[1]
                print(f"{fname} - ver-{ver} - PSNR:{p:.1f} - SSIM:{s:.3f}")
        print()

# -------------------------------------------------------------------------- #
#                          GROUPED STRIP FIGURE BUILD                         #
# -------------------------------------------------------------------------- #
def make_stacked_figure(samples_by_level: Dict[str, List[str]], gen_dirs: List[Path]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folders    = [HE_DIR, GT_DIR] + gen_dirs
    col_labels = ["H&E (orig)", "IHC (GT)"] + make_version_labels(gen_dirs)

    n_levels   = len(HER2_LEVELS)
    n_samples  = len(next(iter(samples_by_level.values())))
    total_rows = 1 + n_levels * (1 + n_samples)
    total_cols = len(folders) + 1  # first col for filename/labels

    fig_height = FIGSIZE[1] * total_rows * 0.3
    fig, axes  = plt.subplots(total_rows, total_cols,
                              figsize=(FIGSIZE[0], fig_height),
                              squeeze=False)

    # column headers
    for j, lbl in enumerate(col_labels, start=1):
        axes[0, j].set_title(lbl, fontsize=12, pad=8)
        axes[0, j].axis("off")
    axes[0, 0].axis("off")

    row = 1
    for lvl in HER2_LEVELS:
        # level label row
        for c in range(total_cols):
            axes[row, c].axis("off")
        axes[row, 0].text(0.5, 0.5, f"HER2 level: {lvl}",
                          ha="center", va="center",
                          fontsize=10, fontweight="bold")
        row += 1

        # sample rows
        for fname in samples_by_level[lvl]:
            axes[row, 0].axis("off")
            axes[row, 0].text(0.5, 0.5, fname,
                              ha="center", va="center", fontsize=8)
            for col, root in enumerate(folders, start=1):
                img = Image.open(root / fname).convert("RGB")
                axes[row, col].imshow(img)
                axes[row, col].axis("off")
            row += 1

    plt.tight_layout()
    out = OUT_DIR / "all_strips_by_level.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out

# ------------------------------------------------------------------------- #
#                     SIDE‑BY‑SIDE COMPARISON STRIP                         #
# ------------------------------------------------------------------------- #
def compare_samples_side_by_side(
        filenames: List[str],
        roots: List[Path],
        col_labels: List[str],
        out_name: str | Path
) -> Path:
    """
    Builds one long PNG where each row = one sample and each column = one
    source (roots[0], roots[1], …).  Each image cell now carries its own
    title so the model / source name is visible on *every* row.

    Parameters
    ----------
    filenames   : list of image filenames to show (same order for every root)
    roots       : list of directories containing the images for each column
                  (e.g. [HE_DIR, GT_DIR, gen_dir1, gen_dir2, …])
    col_labels  : text labels for those columns (same length as roots)
    out_name    : filename to write inside OUT_DIR (e.g. "side_by_side.png")
    """
    assert len(roots) == len(col_labels), "roots and labels length mismatch"

    n_rows = len(filenames)
    n_cols = len(roots) + 1                      # extra left‑hand filename col

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(FIGSIZE[0], n_rows * FIGSIZE[1] * 0.4),
        squeeze=False
    )
    fig.subplots_adjust(wspace=0.0) 
    for r, fname in enumerate(filenames):
        # ── filename cell ───────────────────────────────────────────────── #
        axes[r, 0].axis("off")
        axes[r, 0].text(0.5, 0.5, fname,
                        ha="center", va="center",
                        fontsize=8, fontweight="bold")

        # ── image cells ─────────────────────────────────────────────────── #
        for c, (root, lbl) in enumerate(zip(roots, col_labels), start=1):
            ax  = axes[r, c]
            img = Image.open(root / fname).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(lbl, fontsize=9, pad=0)   # per‑row header

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path



def compare_samples_side_by_side_auto(
    filenames: List[str],
    gen_dirs: List[Path],
    out_name: str
):
    # Put GT right after HE, then all generated versions
    roots  = [HE_DIR, GT_DIR] + gen_dirs
    labels = ["H&E (orig)", "IHC (GT)"] + make_version_labels(gen_dirs)
    return compare_samples_side_by_side(filenames, roots, labels, out_name)

# -------------------------------------------------------------------------- #
#                                    MAIN                                    #
# -------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    samples_by_level = sample_by_level(GEN_DIRS, args.num_samples)
    total = sum(len(v) for v in samples_by_level.values())
    print(f"Selected {total} files ({args.num_samples} per level).")

    compute_metrics(samples_by_level, GEN_DIRS)

    # print("\nBuilding grouped strip figure …")
    # strip_path = make_stacked_figure(samples_by_level, GEN_DIRS)
    # print(f"Saved grouped strips → {strip_path}")

    all_files = [f for lvl in samples_by_level.values() for f in lvl]
    print("\nBuilding tight side-by-side figure …")
    side_path = compare_samples_side_by_side_auto(
        filenames=all_files,
        gen_dirs=GEN_DIRS,
        out_name="all_side_by_side.png"
    )
    print(f"Saved side-by-side comparison → {side_path}")

if __name__ == "__main__":
    main()
