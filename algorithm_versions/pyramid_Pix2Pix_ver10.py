# optimized version of v7 -> on the fly guassian pyramids only + no cropping when downscaling
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import re
from normalize_resized_nocrop_dataset import dataset_mean, dataset_std

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print(torch.cuda.is_available())

# Define paths
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized_nocrop\train"
he_resized_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized_nocrop\train"
checkpoint_dir = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\new_checkpoints\ver_10"

os.makedirs(checkpoint_dir, exist_ok=True)

# DATASET LOADER

class BCIDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.he_images = sorted(
            [f for f in os.listdir(he_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[0])
        )
        self.ihc_images = sorted(
            [f for f in os.listdir(ihc_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[0])
        )
        assert len(self.he_images) == len(self.ihc_images), \
            "Dataset mismatch: Different number of HE and IHC images!"
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform

    def __len__(self):
        return len(self.he_images)

    def __getitem__(self, idx):
        he_file = self.he_images[idx]
        ihc_file = self.ihc_images[idx]
        he = Image.open(os.path.join(self.he_dir, he_file)).convert("RGB")
        ihc = Image.open(os.path.join(self.ihc_dir, ihc_file)).convert("RGB")
        if self.transform:
            he = self.transform(he)
            ihc = self.transform(ihc)
        return he, ihc

# Initialize Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean, std=dataset_std)   

])

dataset = BCIDataset(
    he_dir=he_resized_train_path,
    ihc_dir=ihc_train_path,
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ---------- weight_init.py ----------
def init_weights(net, init_type="normal", gain=0.02):
    """
    He / 'normal' = N(0,0.02) as in pix2pix; 'xavier', 'kaiming', etc. also ok.
    Call after model is created *before* optimisers.
    """
    def init_func(m):                                    # noqa: D401
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1
            or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            # bias
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
            if m.weight is not None:                    # pytorch raises error when affine = False
                nn.init.normal_(m.weight.data, 1.0, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

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
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)   # ⇦ new
        )

        self.res_blocks = nn.Sequential(*[ResNetBlock(256) for _ in range(num_res_blocks)])

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(64),  nn.ReLU(True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Identity()
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
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,128,4,2,1),    nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,256,4,2,1),   nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,512,4,1,1),   nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512,1,4,1,1)
        )
    def forward(self, he, ihc):
        return self.model(torch.cat((he, ihc), 1))

# LOSS FUNCTIONS & OPTIMIZERS
    
criterion_L1 = nn.L1Loss()
criterion_GAN = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GENERATOR & DISCRIMINATOR DEFINITIONS

generator = ResNetGenerator().to(device)
discriminator = PatchDiscriminator().to(device)
init_weights(generator, "normal", 0.02)
init_weights(discriminator, "normal", 0.02)

# MULTI-SCALE LOSS

def gaussian_blur(img: torch.Tensor) -> torch.Tensor:
    """
    3×3 Gaussian with reflect‑padding (matches paper’s scale‑space step).
    img: (N,C,H,W) tensor in [-1,1]
    """
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=img.dtype,
                           device=img.device) / 16.0
    kernel = kernel.expand(img.size(1), 1, 3, 3)        # (C,1,3,3)

    # reflect‑pad *before* convolution; padding=0 in conv
    img = F.pad(img, (1, 1, 1, 1), mode='reflect')       # L,R,T,B
    blurred = F.conv2d(img, kernel, stride=1,
                       padding=0, groups=img.size(1))
    return blurred

def gaussian_pyramid(img, levels=1):
    pyr = [img]
    cur = img
    for _ in range(levels):
        for _ in range(4):            # 4 Gaussian blurs
            cur = gaussian_blur(cur)
        cur = F.avg_pool2d(cur, 2)    # down‑sample once
        pyr.append(cur)
    return pyr

# TRAINING LOOP

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

optimizer_G = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5,0.999))
lr_lambda   = lambda e: 1.0 if e < 25 else 1 - (e-25)/25
sched_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
sched_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

# Load from checkpoint
if latest_checkpoint:
    start_epoch = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, latest_checkpoint)

epochs = 50

for epoch in range(start_epoch, epochs):
    for he, ihc in dataloader:
        he, ihc = he.to(device), ihc.to(device)

        # Discriminator Training
        optimizer_D.zero_grad()

        real_labels = torch.ones_like(discriminator(he, ihc)).to(device)
        fake_labels = torch.zeros_like(real_labels).to(device)

        d_real_loss = criterion_GAN(discriminator(he, ihc), real_labels)
        fake_ihc = generator(he).detach()
        d_fake_loss = criterion_GAN(discriminator(he, fake_ihc), fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()	

        # Generator Training
        optimizer_G.zero_grad()

        fake_ihc  = generator(he)
        g_adv     = criterion_GAN(discriminator(he, fake_ihc), real_labels)

        # Generate fake pyramid (on-the-fly, only for fake images)
        fake_pyr = gaussian_pyramid(fake_ihc, levels=3)[1:]  # levels 1-3
        real_pyr = gaussian_pyramid(ihc, levels=3)[1:]

        # Compute multi-scale loss
        g_multi = sum(F.l1_loss(f, r) for f, r in zip(fake_pyr, real_pyr))

        g_pix     = F.l1_loss(fake_ihc, ihc)
        g_loss    = g_adv + 100*g_pix + g_multi
        g_loss.backward()
        optimizer_G.step()	

    sched_G.step(); sched_D.step()
    print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


    # Every 5 epochs, save images and checkpoint
    if (epoch + 1) % 5 == 0:
        save_checkpoint(epoch + 1, generator, discriminator, optimizer_G, optimizer_D, g_loss.item(), d_loss.item()) 