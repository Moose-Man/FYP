import os
import numpy as np
import cv2
import subprocess
import argparse
from PIL import Image
from tqdm import tqdm  # ✅ Progress Bar
import matplotlib.pyplot as plt
import SimpleITK as sitk

# CONFIGURATION
RESIZE_TO = (128, 128)  # Change to (512, 512) if needed

# Train Paths
he_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\train"
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\train"
he_resized_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train"
ihc_resized_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"
he_registered_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\train"
he_pyramid_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_pyramid\train"

# Test Paths
ihc_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\test"
he_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\test"
he_registered_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\test"
he_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\test"
ihc_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\test"

# Ensure output directories exist
os.makedirs(he_resized_path, exist_ok=True)
os.makedirs(ihc_resized_path, exist_ok=True)
os.makedirs(he_registered_path, exist_ok=True)
os.makedirs(he_pyramid_path, exist_ok=True)
os.makedirs(ihc_resized_test_path, exist_ok=True)
os.makedirs(he_resized_test_path, exist_ok=True)
os.makedirs(he_registered_test_path, exist_ok=True)

# Step 1: Resize Images Before Processing
def resize_and_save(input_path, output_path, size, label):
    """
    Resize images to a consistent size before processing.
    """
    files = [f for f in os.listdir(input_path) if f.endswith(".png") or f.endswith(".jpg")]

    for filename in tqdm(files, desc=f"Resizing {label} images", ncols=80):
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)

        if os.path.exists(output_file):
            continue  # Skip if already resized

        img = Image.open(input_file)
        img = img.resize(size, Image.LANCZOS)
        img.save(output_file)

# Resize H&E and IHC images
resize_and_save(he_train_path, he_resized_path, RESIZE_TO, "H&E")
resize_and_save(ihc_train_path, ihc_resized_path, RESIZE_TO, "IHC")

# Resize IHC test images
resize_and_save(ihc_test_path, ihc_resized_test_path, RESIZE_TO, "IHC Test")
resize_and_save(ihc_test_path, he_resized_test_path, RESIZE_TO, "HE Test")

# Detect and remove black borders from image registration
def remove_black_borders(img, threshold=10):
    """
    Detects and removes black borders in an image by finding the largest non-black region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # Detect non-black areas

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])  # Get bounding box
        cropped_img = img[y:y+h, x:x+w]  # Crop to the detected bounding box
        return cropped_img
    return img  # Return original if no black border found

def is_misaligned_or_blank(img, threshold_std=15, black_threshold=0.15):
    """
    Detects misaligned or blank images based on pixel intensity variance and black region ratio.
    - If variance is too low -> blank/misaligned
    - If black region covers too much -> misaligned
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)  # Measure pixel intensity variation

    # Count black pixels
    black_pixels = np.sum(gray < 5)  # Count pixels close to black (intensity < 5)
    total_pixels = gray.shape[0] * gray.shape[1]
    black_ratio = black_pixels / total_pixels  # Percentage of black area

    print(f"Image StdDev: {std_dev}, Black Area Ratio: {black_ratio:.2f}")

    # Reject image if either condition is met:
    return std_dev < threshold_std or black_ratio > black_threshold

# Step 2: Elastix Registration
elastix_exe = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\elastix\elastix.exe"
elastix_param_file = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\FYP_1\elastix_params.txt"

def register_images(he_path, ihc_path, output_path, param_file):
    """
    Registers H&E images to IHC images using command-line Elastix.
    """
    output_dir = os.path.join(os.path.dirname(output_path), "elastix_output")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        elastix_exe,
        "-f", ihc_path,  # Fixed image (IHC)
        "-m", he_path,  # Moving image (H&E)
        "-out", output_dir,
        "-p", param_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Elastix Output:\n{result.stdout}")
    print(f"Elastix Errors:\n{result.stderr}")

    # Locate Elastix output file
    transformed_img_path = os.path.join(output_dir, "result.0.mhd")

    if not os.path.exists(transformed_img_path):
        print(f"⚠️ Warning: Elastix output missing for {he_path}. Skipping...")
        return

    # Convert `.mhd` to `.png`
    try:
        img_itk = sitk.ReadImage(transformed_img_path)  # Read .mhd file
        img_array = sitk.GetArrayFromImage(img_itk)

        # Ensure correct shape (Elastix might output a single-channel volume)
        if len(img_array.shape) == 3:  
            img_array = img_array[0]  # Extract the first (or only) slice

        # Normalize & convert to 8-bit
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(img)

        # Convert to RGB (if grayscale)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Apply black border removal
        cropped_img = remove_black_borders(img_rgb)

        # Check for misalignment or blank images
        if is_misaligned_or_blank(cropped_img):
            print(f"🛑 Removing misaligned or blank image: {he_path}")
            return  # Skip saving this image
        
        # Resize to match original image dimensions
        cropped_resized_img = cv2.resize(cropped_img, (RESIZE_TO[0], RESIZE_TO[1]), interpolation=cv2.INTER_LINEAR)

        # Resize to match original image dimensions
        img = cv2.resize(img, (RESIZE_TO[0], RESIZE_TO[1]), interpolation=cv2.INTER_LINEAR)

        # Save as PNG
        cv2.imwrite(output_path, cropped_resized_img)

    except Exception as e:
        print(f"⚠️ Error converting {transformed_img_path}: {e}")

# Register images with progress bar (FOR TRAINING)
files = [f for f in os.listdir(he_resized_path)]
for filename in tqdm(files, desc="Registering images", ncols=80):
    he_img_path = os.path.join(he_resized_path, filename)
    ihc_img_path = os.path.join(ihc_resized_path, filename)
    registered_output = os.path.join(he_registered_path, filename)

    if os.path.exists(registered_output):
        continue  # Skip if already registered

    try:
        register_images(he_img_path, ihc_img_path, registered_output, elastix_param_file)
    except Exception as e:
        print(f"⚠️ Warning: Skipping {filename} due to error: {e}")

# Register HE TEST images
files = [f for f in os.listdir(he_test_path)]
for filename in tqdm(files, desc="Registering test images", ncols=80):
    he_img_path = os.path.join(he_resized_test_path, filename)
    ihc_img_path = os.path.join(ihc_resized_test_path, filename)  # Use resized IHC test images
    registered_output = os.path.join(he_registered_test_path, filename)

    if os.path.exists(registered_output):
        continue  # Skip if already registered

    try:
        register_images(he_img_path, ihc_img_path, registered_output, elastix_param_file)
    except Exception as e:
        print(f"⚠️ Warning: Skipping {filename} due to error: {e}")

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
files = [f for f in os.listdir(he_registered_path) if os.path.exists(os.path.join(he_registered_path, f))]

for filename in tqdm(files, desc="Generating Gaussian Pyramid", ncols=80):
    he_image_path = os.path.join(he_registered_path, filename)

    img = cv2.imread(he_image_path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        print(f"⚠️ Warning: Skipping missing or unreadable image: {filename}")
        continue  # ✅ Skip missing images

    print(f"✅ Processing Gaussian Pyramid for {filename}, Shape: {img.shape}")  # Debugging Output

    pyramid = gaussian_pyramid(img, levels=3)

    for i, scaled_img in enumerate(pyramid):
        output_file = os.path.join(he_pyramid_path, f"{filename.split('.')[0]}_scale_{i}.png")
        cv2.imwrite(output_file, scaled_img)

# # Step 4: Display Sample (Before & After)
# sample_he = cv2.imread(os.path.join(he_train_path, '00004_train_1+.png'))
# resized_sample = cv2.imread(os.path.join(he_resized_path, '00004_train_1+.png'))
# registered_sample = cv2.imread(os.path.join(he_registered_path, '00004_train_1+.png'))
# pyramid_sample = cv2.imread(os.path.join(he_pyramid_path, '00004_train_1+_scale_2.png'))

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 4, 1)
# plt.title("Original H&E Image")
# plt.imshow(cv2.cvtColor(sample_he, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 4, 2)
# plt.title("Resized H&E Image")
# plt.imshow(cv2.cvtColor(resized_sample, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 4, 3)
# plt.title("Registered H&E Image")
# plt.imshow(cv2.cvtColor(registered_sample, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 4, 4)
# plt.title("Gaussian Pyramid (Scale 2)")
# plt.imshow(cv2.cvtColor(pyramid_sample, cv2.COLOR_BGR2RGB))

# plt.show()
