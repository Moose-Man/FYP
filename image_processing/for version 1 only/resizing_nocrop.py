import os
from PIL import Image
from tqdm import tqdm

# CONFIGURATION
he_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\train"
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\train"
he_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE\test"
ihc_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC\test"

he_resized_path_nocrop = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized_nocrop"
ihc_resized_path_nocrop = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized_nocrop"

# Create folders if not exist
os.makedirs(os.path.join(he_resized_path_nocrop, "train"), exist_ok=True)
os.makedirs(os.path.join(ihc_resized_path_nocrop, "train"), exist_ok=True)
os.makedirs(os.path.join(he_resized_path_nocrop, "test"), exist_ok=True)
os.makedirs(os.path.join(ihc_resized_path_nocrop, "test"), exist_ok=True)

def resize_images(input_path, output_path, label, size=(128, 128)):
    """
    Resize all images in a folder to a specific size and save them to the output path.
    """
    files = [f for f in os.listdir(input_path) if f.endswith(".png") or f.endswith(".jpg")]
    
    for filename in tqdm(files, desc=f"Resizing {label} images", ncols=80):
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)

        if os.path.exists(output_file):
            continue  # Skip if already processed

        try:
            img = Image.open(input_file).resize(size, Image.LANCZOS)
            img.save(output_file)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Resize all sets directly to 128x128
resize_images(he_train_path, os.path.join(he_resized_path_nocrop, "train"), "H&E Train")
resize_images(ihc_train_path, os.path.join(ihc_resized_path_nocrop, "train"), "IHC Train")
resize_images(he_test_path, os.path.join(he_resized_path_nocrop, "test"), "H&E Test")
resize_images(ihc_test_path, os.path.join(ihc_resized_path_nocrop, "test"), "IHC Test")
