import os
import cv2
import subprocess
import numpy as np
import SimpleITK as sitk
import shutil
from tqdm import tqdm  # ✅ Progress Bar

# CONFIGURATION
he_resized_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train"
ihc_resized_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"
he_registered_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train"
he_pyramid_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_pyramid\train"

ihc_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\test"
he_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\test"
he_registered_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\test"
he_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\test"
ihc_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\test"

# Step 2: Elastix Registration
elastix_exe = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\elastix\elastix.exe"
elastix_param_file = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\FYP_1\elastix_params.txt"

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
    print(f"Elastix Errors:\n{result.stderr}")

    # Locate Elastix output file
    transformed_img_path = os.path.join(output_dir, "result.0.mhd")
    
    if not os.path.exists(transformed_img_path):
        print(f"⚠️ Warning: Elastix output missing for {he_path}. Skipping...")
        return

    # Convert .mhd to .png
    try:
        img_itk = sitk.ReadImage(transformed_img_path)  # Read .mhd file
        img_array = sitk.GetArrayFromImage(img_itk)

        # Ensure correct shape (Elastix might output a single-channel volume)
        if len(img_array.shape) == 3:
            img_array = img_array[0]  # Extract first slice if 3D

        # Normalize & convert to 8-bit
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert to RGB (if grayscale)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Apply black border removal
        cropped_img = remove_black_borders(img_rgb)

        # Check for misalignment or blank images
        if is_misaligned_or_blank(cropped_img):
            print(f"🛑 Removing misaligned or blank image: {he_path}")
            return  # Skip saving this image

        # Resize to match original image dimensions
        cropped_resized_img = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_LINEAR)

        # Delete elastix temp output to save disk space
        shutil.rmtree(output_dir)

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
    ihc_img_path = os.path.join(ihc_resized_test_path, filename)
    registered_output = os.path.join(he_registered_test_path, filename)

    if os.path.exists(registered_output):
        continue  # Skip if already registered

    try:
        register_images(he_img_path, ihc_img_path, registered_output, elastix_param_file)
    except Exception as e:
        print(f"⚠️ Warning: Skipping {filename} due to error: {e}")
