import os
import random
from PIL import Image
from tqdm import tqdm  # ✅ Progress Bar

# CONFIGURATION
he_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\train"
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\train"
he_resized_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train"
ihc_resized_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"
ihc_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\test"
he_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\test"
he_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\test"
ihc_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\test"

# Ensure output directories exist
os.makedirs(he_resized_path, exist_ok=True)
os.makedirs(ihc_resized_path, exist_ok=True)
os.makedirs(ihc_resized_test_path, exist_ok=True)
os.makedirs(he_resized_test_path, exist_ok=True)


# Step 1: Resize Images Before Processing
def resize_and_crop_paired(he_input_path, ihc_input_path, he_output_path, ihc_output_path, size=(256, 256), crop_size=(128, 128)):
    """
    Resize and apply identical random crop to paired H&E and IHC images.
    Ensures that both images in the pair are spatially aligned after processing.
    """
    os.makedirs(he_output_path, exist_ok=True)
    os.makedirs(ihc_output_path, exist_ok=True)

    he_files = sorted([f for f in os.listdir(he_input_path) if f.endswith(".png") or f.endswith(".jpg")])
    ihc_files = sorted([f for f in os.listdir(ihc_input_path) if f.endswith(".png") or f.endswith(".jpg")])

    assert len(he_files) == len(ihc_files), "Mismatch: different number of H&E and IHC images!"
    assert he_files == ihc_files, "Mismatch: H&E and IHC filenames are not aligned!"

    for filename in tqdm(he_files, desc="Resizing and cropping paired images", ncols=80):
        he_in = os.path.join(he_input_path, filename)
        ihc_in = os.path.join(ihc_input_path, filename)
        he_out = os.path.join(he_output_path, filename)
        ihc_out = os.path.join(ihc_output_path, filename)

        if os.path.exists(he_out) and os.path.exists(ihc_out):
            continue  # Skip already processed pairs

        # Load images
        he_img = Image.open(he_in).resize(size, Image.LANCZOS)
        ihc_img = Image.open(ihc_in).resize(size, Image.LANCZOS)

        # Random crop (same coords for both)
        max_left = size[0] - crop_size[0]
        max_top = size[1] - crop_size[1]
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        box = (left, top, left + crop_size[0], top + crop_size[1])

        he_crop = he_img.crop(box)
        ihc_crop = ihc_img.crop(box)

        # Save
        he_crop.save(he_out)
        ihc_crop.save(ihc_out)


# Step 1B: Resize test images to 256x256 and center crop to 128x128
def resize_and_center_crop(input_path, output_path, label):
    """
    Resizes test images to 256x256, then applies center crop to 128x128.
    This ensures deterministic preprocessing for evaluation.
    """
    files = [f for f in os.listdir(input_path) if f.endswith(".png") or f.endswith(".jpg")]

    for filename in tqdm(files, desc=f"Center-cropping {label} test images", ncols=80):
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)

        if os.path.exists(output_file):
            continue  # Skip if already processed

        img = Image.open(input_file)

        # Step 1: Resize to 256x256
        img = img.resize((256, 256), Image.LANCZOS)

        # Step 2: Center crop to 128x128
        left = (256 - 128) // 2  # = 64
        top = (256 - 128) // 2   # = 64
        right = left + 128
        bottom = top + 128
        img = img.crop((left, top, right, bottom))

        img.save(output_file)


# Resize H&E and IHC images
resize_and_crop_paired(
    he_input_path=he_train_path,
    ihc_input_path=ihc_train_path,
    he_output_path=he_resized_path,
    ihc_output_path=ihc_resized_path,
    size=(256, 256),
    crop_size=(128, 128)
)

# Apply center crop preprocessing for test images
resize_and_center_crop(ihc_test_path, ihc_resized_test_path, "IHC")
resize_and_center_crop(he_test_path, he_resized_test_path, "H&E")
