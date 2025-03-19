import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Paths
checkpoint_dir = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\checkpoints"
he_registered_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\test"
ihc_resized_test_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\test"
save_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results"
os.makedirs(save_path, exist_ok=True)

# ✅ Dataset loader
class BCIDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.he_images = sorted(
            [f for f in os.listdir(he_dir) if f.endswith(".png")], key=lambda x: int(x.split("_")[0])
        )
        self.ihc_images = sorted(
            [f for f in os.listdir(ihc_dir) if f.endswith(".png")], key=lambda x: int(x.split("_")[0])
        )

        assert len(self.he_images) == len(self.ihc_images), "Dataset mismatch: Different number of HE and IHC images!"
 
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform

    def __len__(self):
        return len(self.he_images)  

    def __getitem__(self, idx):
        he_path = os.path.join(self.he_dir, self.he_images[idx])
        ihc_path = os.path.join(self.ihc_dir, self.ihc_images[idx])

        he_image = Image.open(he_path).convert("RGB")
        ihc_image = Image.open(ihc_path).convert("RGB")

        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)

        return he_image, ihc_image

# ✅ Find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest generator checkpoint."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))

    if not checkpoint_files:
        print("⚠️ No checkpoint found. Make sure you've trained the model.")
        return None

    def extract_epoch(filename):
        match = re.search(r"checkpoint_epoch_(\d+).pth", filename)
        return int(match.group(1)) if match else 0

    checkpoint_files.sort(key=extract_epoch, reverse=True)
    latest_checkpoint = checkpoint_files[0]

    print(f"✅ Loading generator checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# ✅ Load the trained generator model
def load_generator(generator, checkpoint_path):
    """Loads the trained generator model."""
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()  # Set to evaluation mode
    return generator

# ✅ Define ResNet Generator
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_res_blocks=9):
        super(ResNetGenerator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResNetBlock(128) for _ in range(num_res_blocks)])

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.res_blocks(x)
        x = self.upsampling(x)
        x = self.final(x)
        return x

# ✅ Run generator on test images & evaluate
def test_generator(generator, test_dataloader, save_path):
    """Runs the trained generator on test H&E images and evaluates results."""
    os.makedirs(save_path, exist_ok=True)

    psnr_scores = []
    ssim_scores = []
    l1_losses = []

    criterion_L1 = nn.L1Loss()

    # Get filenames of IHC images for correct naming
    ihc_filenames = sorted([f for f in os.listdir(ihc_resized_test_path) if f.endswith(".png")])

    with torch.no_grad():
        for i, (he_image, ihc_real) in enumerate(test_dataloader):
            he_image, ihc_real = he_image.to(device), ihc_real.to(device)

            # Generate IHC image
            ihc_fake = generator(he_image)

            # Convert `Tanh` output from [-1,1] to [0,1]
            ihc_fake = (ihc_fake + 1) / 2
            ihc_real = (ihc_real + 1) / 2

            # Convert tensors to numpy (H, W, C) & scale to 0-255
            ihc_fake_np = (ihc_fake.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)
            ihc_real_np = (ihc_real.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)

            # Compute evaluation metrics
            psnr_value = psnr(ihc_real_np, ihc_fake_np, data_range=255)
            ssim_value = ssim(ihc_real_np, ihc_fake_np, channel_axis=2, data_range=255)
            l1_loss = criterion_L1(ihc_fake, ihc_real).item()

            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)
            l1_losses.append(l1_loss)

            # Save generated images
            plt.imsave(os.path.join(save_path, f"{ihc_filenames[i]}"), ihc_fake_np)

            print(f"Test {i+1}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}, L1 Loss={l1_loss:.6f}")

    print(f"\n✅ Average PSNR: {np.mean(psnr_scores):.4f}")
    print(f"✅ Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"✅ Average L1 Loss: {np.mean(l1_losses):.6f}")

# ✅ Run Testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

generator = ResNetGenerator().to(device)
generator = load_generator(generator, latest_checkpoint)

test_dataset = BCIDataset(he_dir=he_registered_test_path, ihc_dir=ihc_resized_test_path, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_generator(generator, test_dataloader, save_path)
