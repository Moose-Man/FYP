import os, cv2
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION  –– edit only these three paths / params
# ------------------------------------------------------------------
ihc_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"
ihc_pyramid_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_pyramid\train"
n_scales         = 3                       # S1..S3 (S0=input)

# ------------------------------------------------------------------
# Ensure output directory exists
# ------------------------------------------------------------------
os.makedirs(ihc_pyramid_path, exist_ok=True)

# ------------------------------------------------------------------
# Paper‑style Gaussian pyramid
# ------------------------------------------------------------------
def gaussian_pyramid_paper(img_bgr, n_scales=3):
    """
    Replicates the 'pyramid pix2pix' scale‑space step:
      • 3×3 Gaussian (σ=1) applied FOUR times
      • then down‑sample (½) once  → next octave first layer
    Returns: list [S0, S1, S2, ...] where S0 = original.
    """
    g_kernel = (3, 3)          # OpenCV wants (w,h)
    sigma    = 1
    scales   = [img_bgr]       # S0

    current = img_bgr.copy()
    for _ in range(n_scales):
        # 4 successive blurs
        for _ in range(4):
            current = cv2.GaussianBlur(current, g_kernel, sigmaX=sigma, sigmaY=sigma,
                                       borderType=cv2.BORDER_REFLECT101)
        # down‑sample by factor 2
        current = cv2.pyrDown(current, borderType=cv2.BORDER_REFLECT101)
        scales.append(current)

    return scales  # length = n_scales+1

# ------------------------------------------------------------------
# Iterate through dataset and write images
# ------------------------------------------------------------------
files = [f for f in os.listdir(ihc_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for filename in tqdm(files, desc="Building Gaussian Pyramid", ncols=80):
    src_path = os.path.join(ihc_path, filename)
    img      = cv2.imread(src_path, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        print(f"⚠️  Skipping unreadable file: {filename}")
        continue

    # Build pyramid (add/remove center‑crop here ONLY if you also did it at train time)
    pyramid = gaussian_pyramid_paper(img, n_scales)

    base_name, _ = os.path.splitext(filename)
    for i, level_img in enumerate(pyramid):
        out_path = os.path.join(ihc_pyramid_path, f"{base_name}_scale_{i}.png")
        cv2.imwrite(out_path, level_img)

