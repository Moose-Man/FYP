import os
import cv2
from tqdm import tqdm  # ✅ Progress Bar

# CONFIGURATION
he_registered_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\train"
he_pyramid_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_pyramid\train"

# Ensure output directory exists
os.makedirs(he_pyramid_path, exist_ok=True)


# Step 3: Gaussian Pyramid Processing (Multi-Scale Representation)
def gaussian_pyramid(img, levels=3):
    """
    Generates a Gaussian pyramid of an image with multiple scales.
    """
    pyramid = [img]
    for _ in range(levels):
        img = cv2.GaussianBlur(img, (5, 5), 1)  # Apply Gaussian filter
        img = cv2.pyrDown(img)  # Downsample
        pyramid.append(img)
    return pyramid


# Apply multi-scale transformation with progress bar
files = [f for f in os.listdir(he_registered_path)]

for filename in tqdm(files, desc="Generating Gaussian Pyramid", ncols=80):
    he_image_path = os.path.join(he_registered_path, filename)

    # Skip if all pyramid levels already exist
    if all(os.path.exists(os.path.join(he_pyramid_path, f"{filename.split('.')[0]}_scale_{i}.png")) for i in range(4)):
        continue  # Skip already processed pyramid

    img = cv2.imread(he_image_path, cv2.IMREAD_COLOR)

    if img is None or img.size == 0:
        print(f"⚠️ Warning: Skipping missing or unreadable image: {filename}")
        continue  # ✅ Skip missing images

    # print(f"✅ Processing Gaussian Pyramid for {filename}, Shape: {img.shape}")  # Debugging Output

    # Apply center crop to 128x128 before pyramid generation
    h, w, _ = img.shape
    top = (h - 128) // 2
    left = (w - 128) // 2
    img = img[top:top+128, left:left+128]

    pyramid = gaussian_pyramid(img, levels=3)

    for i, scaled_img in enumerate(pyramid):
        output_file = os.path.join(he_pyramid_path, f"{filename.split('.')[0]}_scale_{i}.png")
        cv2.imwrite(output_file, scaled_img)
