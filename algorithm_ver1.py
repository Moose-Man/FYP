import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import staintools
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(torch.cuda.is_available())

# Define paths to dataset
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\BCI_dataset\IHC\train"
he_normalized_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\BCI_dataset\HE_normalized\train"

# DATASET LOADER

class BCIDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.he_images = sorted(os.listdir(he_dir))  # Get list of HE images
        self.ihc_images = sorted(os.listdir(ihc_dir))  # Get corresponding IHC images
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform

    def __len__(self):
        return len(self.he_images)  # Number of samples

    def __getitem__(self, idx):
        he_path = os.path.join(self.he_dir, self.he_images[idx])
        ihc_path = os.path.join(self.ihc_dir, self.ihc_images[idx])

        # Load images
        he_image = Image.open(he_path).convert("RGB")
        ihc_image = Image.open(ihc_path).convert("RGB")

        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)

        return he_image, ihc_image

# INITIALIZE DATALOADER

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = BCIDataset(he_dir=he_normalized_train_path, ihc_dir=ihc_train_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# MODEL IMPLEMENTATION BLOCK (temporarily using this structure)

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # Output scaled between 0-1
        return x

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = torch.sigmoid(self.conv3(x))  # Binary classification output
        return x

# LOSS FUNCTIONS/OPTIMIZERS BLOCK

criterion_L1 = nn.L1Loss()
criterion_GAN = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Debugging info

# Instantiate models and move them to GPU
generator = UNetGenerator()
discriminator = PatchDiscriminator()
generator.to(device)
discriminator.to(device)

# # Check where the models are located
# print(next(generator.parameters()).device)  # Should print "cuda:0"
# print(next(discriminator.parameters()).device)  # Should print "cuda:0"

# Initialize optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# TRAINING LOOP

from torch.amp import autocast, GradScaler

scaler = torch.amp.GradScaler()  # Handles automatic loss scaling

epochs = 50

def load_checkpoint(filepath, generator, discriminator, optimizer_G, optimizer_D, scaler):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"✅ Resumed from {filepath}, starting at epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    return 0  # Start fresh if no checkpoint found

# Automatically find the latest checkpoint
checkpoint_files = sorted(glob.glob("checkpoint_epoch_*.pth"))  # List all checkpoint files
latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None  # Get the most recent one

start_epoch = load_checkpoint(latest_checkpoint, generator, discriminator, optimizer_G, optimizer_D, scaler)

for epoch in range(start_epoch, epochs):

    generator.train()
    discriminator.train()

    for i, (he, ihc) in enumerate(dataloader):
        he, ihc = he.to(device), ihc.to(device)  # Move batch to GPU

        ### 1️⃣ Train Discriminator ###
        optimizer_D.zero_grad()

        with torch.no_grad():
            sample_output = discriminator(torch.cat((he[:1], ihc[:1]), 1))  # Get shape from a test batch
            label_shape = sample_output.shape  # Ensure label shape matches discriminator output

        real_labels = torch.ones(label_shape).to(device) # PatchGAN labels
        fake_labels = torch.zeros(label_shape).to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # Enable AMP
            real_output = discriminator(torch.cat((he, ihc), 1))  # D(HE, Real IHC)
            d_loss_real = criterion_GAN(real_output, real_labels)

            fake_ihc = generator(he)
            fake_output = discriminator(torch.cat((he, fake_ihc.detach()), 1))  # D(HE, Fake IHC)
            d_loss_fake = criterion_GAN(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2  # Total D loss

        scaler.scale(d_loss).backward()  # Scale loss before backpropagation
        scaler.step(optimizer_D)  # Step optimizer
        scaler.update()  # Update scaler

        ### 2️⃣ Train Generator ###
        optimizer_G.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # Enable AMP
            fake_output = discriminator(torch.cat((he, fake_ihc), 1))  # D(HE, Fake IHC)
            g_loss_adv = criterion_GAN(fake_output, real_labels)

            g_loss_l1 = criterion_L1(fake_ihc, ihc)  # L1 Loss for pixel similarity
            g_loss = g_loss_adv + (100 * g_loss_l1)  # Total G loss

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

    # save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, f"checkpoint_epoch_{epoch + 1}.pth")
        print(f"✅ Checkpoint saved at epoch {epoch + 1}")

    print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")