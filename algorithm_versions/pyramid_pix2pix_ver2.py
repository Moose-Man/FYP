import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.spectral_norm as SN
from PIL import Image
import matplotlib.pyplot as plt
import glob
import re
from torchvision.models import vgg19, VGG19_Weights
from torch.amp import autocast, GradScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print(torch.cuda.is_available())

# Define paths
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"
he_registered_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_registered\train"
checkpoint_dir = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\checkpoints\ver_2\ver_2_test"
os.makedirs(checkpoint_dir, exist_ok=True)

# DATASET LOADER

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

# Initialize Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = BCIDataset(he_dir=he_registered_train_path, ihc_dir=ihc_train_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# RESNET-9 BLOCKS GENERATOR

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

# PATCHGAN DISCRIMINATOR

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            SN(nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))  
        )

    def forward(self, he, ihc):
        x = torch.cat((he, ihc), 1)
        return self.model(x)

# LOSS FUNCTIONS & OPTIMIZERS

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16]  # Extract features from early VGG layers
        self.vgg = nn.Sequential(*[layer for layer in vgg])  # Use only feature extractor
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        return F.l1_loss(fake_features, real_features)  # Perceptual loss
    
criterion_L1 = nn.L1Loss()
criterion_GAN = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = ResNetGenerator().to(device)
discriminator = PatchDiscriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# MULTI-SCALE LOSS

def gaussian_pyramid(img, levels=3):
    pyramid = [img]
    for _ in range(levels):
        img = F.avg_pool2d(img, kernel_size=2)
        pyramid.append(img)
    return pyramid

# TRAINING LOOP

def save_sample_images(epoch, he, fake_ihc, ihc):
    he = he.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    fake_ihc = fake_ihc.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    ihc = ihc.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]

    # Rescale from [-1, 1] to [0, 1] and convert to float32 for plotting
    he = ((he + 1) / 2).clip(0, 1).astype('float32')
    fake_ihc = ((fake_ihc + 1) / 2).clip(0, 1).astype('float32')
    ihc = ((ihc + 1) / 2).clip(0, 1).astype('float32')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input H&E")
    plt.imshow(he)

    plt.subplot(1, 3, 2)
    plt.title(f"Generated IHC (Epoch {epoch})")
    plt.imshow(fake_ihc)

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth IHC")
    plt.imshow(ihc)

    output_dir = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\checkpoints\ver_2"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_output.png"))

    plt.pause(10)
    plt.close()

def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, g_loss, d_loss):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved at {checkpoint_path}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ No checkpoint found at {checkpoint_path}, starting from scratch.")
        return 0  # Start from epoch 0

    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    print(f"✅ Loaded checkpoint from {checkpoint_path} (Epoch {checkpoint['epoch']})")
    return checkpoint['epoch']  # Return last saved epoch

def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint in the given directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))

    if not checkpoint_files:
        print("⚠️ No checkpoint found, starting from scratch.")
        return None  # No checkpoint found

    # Sort checkpoints by epoch number (extract epoch from filename)

    def extract_epoch(filename):
        match = re.search(r"checkpoint_epoch_(\d+).pth", filename)
        return int(match.group(1)) if match else 0

    checkpoint_files.sort(key=extract_epoch, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    
    print(f"✅ Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# Look for the latest checkpoint 
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
start_epoch = 0  # Default starting epoch

# Load from checkpoint
if latest_checkpoint:
    start_epoch = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, latest_checkpoint)

epochs = 50

perceptual_loss = VGGLoss().to(device)

scaler = GradScaler(device='cuda')

for epoch in range(start_epoch, epochs):
    for he_img, ihc in dataloader:
        ihc = ihc.to(device)
        he_img = he_img.to(device)

        # Discriminator training
        with autocast(device_type='cuda'):
            fake_ihc = generator(he_img)
            real_labels = torch.ones_like(discriminator(he_img, ihc))
            fake_labels = torch.zeros_like(real_labels)

            d_real_loss = criterion_GAN(discriminator(he_img, ihc), real_labels)
            d_fake_loss = criterion_GAN(discriminator(he_img, fake_ihc.detach()), fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2

        optimizer_D.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)

        # Generator training
        with autocast(device_type='cuda'):
            fake_ihc = generator(he_img)
            g_adv_loss = criterion_GAN(discriminator(he_img, fake_ihc), real_labels)

            # Now generate pyramid for both fake and real
            fake_pyramid = gaussian_pyramid(fake_ihc, levels=3)
            real_pyramid = gaussian_pyramid(he_img, levels=3)

            g_multi_loss = sum(F.l1_loss(f, r) for f, r in zip(fake_pyramid, real_pyramid))
            perceptual = perceptual_loss(fake_ihc, ihc)

            g_loss = g_adv_loss + 100 * g_multi_loss + 10 * perceptual

        optimizer_G.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

    print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    if (epoch + 1) % 5 == 0:
        save_sample_images(epoch + 1, he_img, fake_ihc, ihc)
        save_checkpoint(epoch + 1, generator, discriminator, optimizer_G, optimizer_D, g_loss.item(), d_loss.item())