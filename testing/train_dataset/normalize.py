import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set your H&E and IHC training image directories here
HE_TRAIN_DIR_1 = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train"
HE_TRAIN_DIR_2 = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\train"
IHC_TRAIN_DIR = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"

def compute_mean_std(directories):
    image_files = []
    for directory in directories:
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.endswith(".png") or f.endswith(".jpg")]
        image_files.extend(files)

    mean = np.zeros(3)
    std = np.zeros(3)
    pixel_count = 0

    for file in tqdm(image_files, desc="Calculating mean and std"):
        img = Image.open(file).convert("RGB")
        img = np.array(img) / 255.0  # Scale to [0, 1]

        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
        pixel_count += 1

    mean /= pixel_count
    std /= pixel_count

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    mean, std = compute_mean_std([HE_TRAIN_DIR_2, IHC_TRAIN_DIR])
    print("Dataset Mean:", mean)
    print("Dataset Std:", std)
else:
    # Allows importing the values when used as a module
    mean, std = compute_mean_std([HE_TRAIN_DIR_2, IHC_TRAIN_DIR])
    dataset_mean = mean
    dataset_std = std

print(compute_mean_std)
