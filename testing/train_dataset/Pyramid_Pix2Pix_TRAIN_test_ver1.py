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
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

epoch_chosen = 50

checkpoint_dir = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\new_checkpoints\ver_1"

he_registered_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered_nocrop\train"

# for testing on test dataset
ihc_resized_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized_nocrop\train"

# save path for generated images for test dataset
image_save_path_train = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\ver_1"

# save path for test dataset graphs
graph_save_path_train = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\test_results\train_dataset\metric_graphs\ver_1"

os.makedirs(image_save_path_train, exist_ok=True)
os.makedirs(graph_save_path_train, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# === Dataset loader ===
class BCIDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.he_images = sorted(
            [f for f in os.listdir(he_dir) if f.endswith(".png")], key=lambda x: int(x.split("_")[0])
        )
        self.ihc_images = sorted(
            [f for f in os.listdir(ihc_dir) if f.endswith(".png")], key=lambda x: int(x.split("_")[0])
        )

        assert len(self.he_images) == len(self.ihc_images), "Mismatch: HE and IHC counts differ!"
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

# === ResNet Generator ===
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_blocks=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 1, 3, padding_mode='reflect'),
            nn.InstanceNorm2d(64), nn.ReLU(True)
        )
        self.downsampling = nn.Sequential(      # 64 → 128 → 256
            nn.Conv2d(64, 128, 3, 2, 1),  nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(*[ResNetBlock(256) for _ in range(n_blocks)])
        self.upsampling = nn.Sequential(        # 256 → 128 → 64
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),  nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, 7, 1, 3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.res_blocks(x)
        x = self.upsampling(x)
        return self.final(x)


# === Checkpoint Manager ===
def get_all_checkpoints(checkpoint_dir):
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    def extract_epoch(f): return int(re.search(r"(\d+)", os.path.basename(f)).group(1))
    files = sorted(files, key=extract_epoch)
    return [(extract_epoch(f), f) for f in files if extract_epoch(f) % 5 == 0]

def load_generator(generator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

# === Evaluation ===
def test_generator(generator, dataloader):
    psnr_scores, ssim_scores, l1_losses = [], [], []
    criterion_L1 = nn.L1Loss()

    with torch.no_grad():
        for he_image, ihc_real in dataloader:
            he_image, ihc_real = he_image.to(device), ihc_real.to(device)
            ihc_fake = generator(he_image)

            ihc_fake = (ihc_fake + 1) / 2
            ihc_real = (ihc_real + 1) / 2

            ihc_fake_np = (ihc_fake.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)
            ihc_real_np = (ihc_real.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)

            psnr_scores.append(psnr(ihc_real_np, ihc_fake_np, data_range=255))
            ssim_scores.append(ssim(ihc_real_np, ihc_fake_np, channel_axis=2, data_range=255))
            l1_losses.append(criterion_L1(ihc_fake, ihc_real).item())

    return np.mean(psnr_scores), np.mean(ssim_scores), np.mean(l1_losses)

# === Graphing ===
def plot_metrics(metrics_dict, save_path):
    """
    Plots PSNR, SSIM, and L1 Loss separately over epochs and saves them as images.

    Args:
        metrics_dict (dict): Dictionary containing 'epochs', 'psnr', 'ssim', and 'l1'.
        save_path (str): Directory where plots should be saved.
    """
    epochs = metrics_dict['epochs']
    psnr = metrics_dict['psnr']
    ssim = metrics_dict['ssim']
    l1 = metrics_dict['l1']

    # Plot PSNR
    plt.figure()
    plt.plot(epochs, psnr, marker='o', color='blue')
    plt.title("PSNR Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.xlim(0, 50)
    plt.ylim(10, 30)
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "psnr_over_epochs.png"))
    plt.close()

    # Plot SSIM
    plt.figure()
    plt.plot(epochs, ssim, marker='o', color='orange')
    plt.title("SSIM Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "ssim_over_epochs.png"))
    plt.close()

    # Plot L1 Loss
    plt.figure()
    plt.plot(epochs, l1, marker='o', color='green')
    plt.title("L1 Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.xlim(0, 50)
    plt.ylim(0, 1)  
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "l1_loss_over_epochs.png"))
    plt.close()

    print("✅ Saved PSNR, SSIM, and L1 Loss plots to", save_path)

# === Save Metrics to Text File ===
def save_metrics_to_txt(metrics_dict, save_path):
    """
    Saves PSNR, SSIM, and L1 Loss values to a text file.
    
    Args:
        metrics_dict (dict): Contains 'epochs', 'psnr', 'ssim', and 'l1'.
        save_path (str): Directory where text file should be saved.
    """
    txt_path = os.path.join(save_path, "metrics_summary.txt")
    with open(txt_path, "w") as f:
        for epoch, psnr_val, ssim_val, l1_val in zip(
            metrics_dict["epochs"], metrics_dict["psnr"], metrics_dict["ssim"], metrics_dict["l1"]
        ):
            f.write(f"Epoch {epoch}:\n")
            f.write(f"    PSNR: {psnr_val:.2f}\n")
            f.write(f"    SSIM: {ssim_val:.4f}\n")
            f.write(f"    L1 Loss: {l1_val:.4f}\n\n")
    print(f"📝 Saved metric summary to {txt_path}")

# === MAIN ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)   # maps 0‑1 → −1‑1
])
test_dataset = BCIDataset(he_registered_train_path, ihc_resized_train_path, transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

checkpoints = get_all_checkpoints(checkpoint_dir)

epochs, psnr_list, ssim_list, l1_list = [], [], [], []

for epoch, ckpt_path in tqdm(checkpoints, desc="🔬 Testing Checkpoints", unit="ckpt"):
    print(f"🔍 Testing checkpoint @ epoch {epoch}")
    generator = ResNetGenerator().to(device)
    generator = load_generator(generator, ckpt_path)

    psnr_val, ssim_val, l1_val = test_generator(generator, test_dataloader)

    print(f"- epoch {epoch}:")
    print(f"    - Average PSNR: {psnr_val:.2f}")
    print(f"    - Average SSIM: {ssim_val:.4f}")
    print(f"    - Average L1 Loss: {l1_val:.4f}")        
    
    epochs.append(epoch)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    l1_list.append(l1_val)

metrics = {
    "epochs": epochs,
    "psnr": psnr_list,
    "ssim": ssim_list,
    "l1": l1_list
}

plot_metrics(metrics, graph_save_path_train)
save_metrics_to_txt(metrics, graph_save_path_train)

def save_generated_images_for_epoch(generator, dataloader, epoch, output_dir):
    """
    Generates and saves IHC predictions for a given epoch.
    
    Args:
        generator: Trained generator model (already loaded with weights).
        dataloader: Test dataloader (BCIDataset).
        epoch (int): Epoch number (used for naming the folder).
        output_dir (str): Base directory to save generated images.
    """
    generator.eval()
    device = next(generator.parameters()).device

    save_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"📸 Saving generated images to: {save_dir}")

    ihc_filenames = sorted([
        f for f in os.listdir(ihc_resized_train_path) if f.endswith(".png")
    ])

    with torch.no_grad():
        for i, (he_image, _) in enumerate(dataloader):
            he_image = he_image.to(device)
            ihc_fake = generator(he_image)
            ihc_fake = (ihc_fake + 1) / 2  # Tanh → [0, 1]
            ihc_fake_np = (ihc_fake.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8)

            image_filename = ihc_filenames[i] if i < len(ihc_filenames) else f"generated_{i}.png"
            save_path = os.path.join(save_dir, image_filename)

            plt.imsave(save_path, ihc_fake_np)

    print("✅ Finished saving generated images.")

# ✅ Return a specific checkpoint or the latest one
def return_checkpoint(checkpoint_dir, epoch=None):
    """
    Returns the checkpoint path for a specific epoch or the latest checkpoint.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files.
        epoch (int, optional): The epoch number of the desired checkpoint. If None, returns the latest checkpoint.
    
    Returns:
        str: Path to the checkpoint file.
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))

    if not checkpoint_files:
        print("⚠️ No checkpoint found. Make sure you've trained the model.")
        return None

    def extract_epoch(filename):
        match = re.search(r"checkpoint_epoch_(\d+).pth", filename)
        return int(match.group(1)) if match else None

    checkpoint_files.sort(key=extract_epoch, reverse=True)  # Sort by epoch in descending order

    if epoch is None:
        # Return the latest checkpoint
        latest_checkpoint = checkpoint_files[0]
        print(f"✅ Loading latest generator checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    # Look for a specific epoch checkpoint
    for checkpoint in checkpoint_files:
        if f"checkpoint_epoch_{epoch}.pth" in checkpoint:
            print(f"✅ Loading generator checkpoint for epoch {epoch}: {checkpoint}")
            return checkpoint

    print(f"⚠️ Checkpoint for epoch {epoch} not found.")
    return None

checkpoint = return_checkpoint(checkpoint_dir, epoch=epoch_chosen)
generator = ResNetGenerator().to(device)
generator = load_generator(generator, checkpoint)

save_generated_images_for_epoch(
    generator=generator,
    dataloader=test_dataloader,
    epoch=epoch_chosen,
    output_dir=image_save_path_train
)
