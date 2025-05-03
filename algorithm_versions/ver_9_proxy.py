# version 9 - involves patchNCE contrastive loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import re
from normalize_resized_crop_dataset import dataset_mean, dataset_std
import matplotlib.pyplot as plt

# Define paths
ihc_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\IHC_resized\train"
he_resized_train_path = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\HE_resized\train"
checkpoint_dir = r"C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\new_checkpoints\ver_9_proxy"

lambda_reg_values = [4]
lambda_nce_values = [0.5]
temp_values      = [0.085]
num_negatives_values = [64, 128, 256, 512]

# === PatchNCE: Encoder + Loss ===
class PatchEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # downsample by 2 → 64×64
            nn.Conv2d(in_ch,   feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # downsample by 2 → 32×32
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # final embed
            nn.Conv2d(feat_dim, feat_dim, 3, stride=1, padding=1),
        )
    def forward(self, x):
        # x: (N,3,128,128) → feat: (N,256,32,32)
        return self.net(x)

class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07, num_negatives=256):
        super().__init__()
        self.temperature   = temperature
        self.num_negatives = num_negatives
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        N, C, H, W = feat_q.shape
        M = N * H * W
        q = feat_q.permute(0,2,3,1).reshape(M, C)
        k = feat_k.permute(0,2,3,1).reshape(M, C)

        # positive logits
        l_pos = (q * k).sum(dim=1, keepdim=True) / self.temperature

        # sample K negatives
        idx_neg = torch.randperm(M, device=q.device)[: self.num_negatives]
        k_neg   = k[idx_neg]                        # (K, C)
        l_neg   = q @ k_neg.t() / self.temperature  # (M, K)

        logits = torch.cat([l_pos, l_neg], dim=1)   # (M, 1+K)
        labels = torch.zeros(M, dtype=torch.long, device=q.device)
        return self.cross_entropy(logits, labels)

def show_tensor_image(tensor):
    """Convert tensor to displayable format"""
    image = tensor.cpu().detach().clone()
    image = image * torch.tensor(dataset_std).view(3, 1, 1) + torch.tensor(dataset_mean).view(3, 1, 1)
    image = torch.clamp(image, 0, 1)
    return image.permute(1, 2, 0).numpy()

def visualize_samples(generator, stn, dataloader, num_samples=3):
    generator.eval()
    stn.eval()

    with torch.no_grad():
        for i, (he, ihc) in enumerate(dataloader):
            if i >= num_samples:
                break
            he, ihc = he.to(device), ihc.to(device)

            warped_ihc, _ = stn(he, ihc)
            fake_ihc = generator(he)

            he_img = show_tensor_image(he[0])
            ihc_img = show_tensor_image(ihc[0])
            warped_img = show_tensor_image(warped_ihc[0])
            fake_img = show_tensor_image(fake_ihc[0])

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(he_img)
            axs[0].set_title("H&E (Input)")
            axs[1].imshow(ihc_img)
            axs[1].set_title("Real IHC")
            axs[2].imshow(warped_img)
            axs[2].set_title("Warped IHC")
            axs[3].imshow(fake_img)
            axs[3].set_title("Generated IHC")

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print(torch.cuda.is_available())

os.makedirs(checkpoint_dir, exist_ok=True)

# Spatial Transformer Network 

class STN(nn.Module):
    def __init__(self):
        super().__init__()
        # 6→32→64 conv layers, then FC to 6 params
        self.loc_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),     nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # initialize to identity transform
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 6)
        )
        # init fc_loc to [1,0,0, 0,1,0]
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0, 0,1,0], dtype=torch.float))
    def forward(self, he, ihc):
        x = torch.cat([he, ihc], dim=1)               # (N,6,H,W)
        xs = self.loc_net(x).view(x.size(0), -1)      # (N,64)
        theta = self.fc_loc(xs).view(-1, 2, 3)        # (N,2,3)
        grid  = F.affine_grid(theta, ihc.size(), align_corners=True)
        warped_ihc = F.grid_sample(ihc, grid, align_corners=True)
        return warped_ihc, theta

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
        he_filename = self.he_images[idx]
        ihc_filename = self.ihc_images[idx]

        he_image = Image.open(os.path.join(self.he_dir, he_filename)).convert("RGB")
        ihc_image = Image.open(os.path.join(self.ihc_dir, ihc_filename)).convert("RGB")

        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)

        return he_image, ihc_image

# Initialize Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean, std=dataset_std)   

])

subset = 100
dataset = torch.utils.data.Subset(
    BCIDataset(
        he_dir=he_resized_train_path,
        ihc_dir=ihc_train_path,
        transform=transform
    ),
    list(range(subset))
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
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

stn = STN().to(device)
# 1) apply your normal init to everything
init_weights(stn, "normal", 0.02)

# 2) now *force* the last Linear layer back to identity
with torch.no_grad():
    # fc_loc is nn.Sequential([Linear(64→32), ReLU, Linear(32→6)])
    last_fc: nn.Linear = stn.fc_loc[2]
    last_fc.weight.zero_()   # zero out the 6×32 weight matrix
    # set bias = [1,0,0, 0,1,0]
    last_fc.bias.copy_(torch.tensor([1.,0.,0., 0.,1.,0.], device=device))

# GENERATOR & DISCRIMINATOR DEFINITIONS

generator = ResNetGenerator().to(device)
discriminator = PatchDiscriminator().to(device)
init_weights(generator, "normal", 0.02)
init_weights(discriminator, "normal", 0.02)

# PatchEncoder instantiation
encoder       = PatchEncoder(in_ch=3, feat_dim=256).to(device)
patchnce_loss = PatchNCELoss(temperature=0.07).to(device)
lambda_nce    = 1.0   # proxy weight you’ll sweep over later

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
        'stn_state_dict': stn.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss
    }
    checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved at {checkpoint_path}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, stn, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ No checkpoint found at {checkpoint_path}, starting from scratch.")
        return 0  # Start from epoch 0

    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    stn.load_state_dict(checkpoint['stn_state_dict'])

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

optimizer_G = torch.optim.Adam(list(generator.parameters()) + list(stn.parameters()) + list(encoder.parameters()), 2e-4, betas=(0.5,0.999))
optimizer_G = torch.optim.Adam(list(generator.parameters()) + list(stn.parameters()) + list(encoder.parameters()), 2e-4, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5,0.999))
lr_lambda   = lambda e: 1.0 if e < 25 else 1 - (e-25)/25
sched_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
sched_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

# Load from checkpoint
if latest_checkpoint:
    start_epoch = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, stn, latest_checkpoint)

epochs  = 5

for lambda_reg in lambda_reg_values:
    for lambda_nce in lambda_nce_values:
        for temperature in temp_values:
            for num_negatives in num_negatives_values:
                # update temperature
                patchnce_loss = PatchNCELoss(
                    temperature=temperature,
                    num_negatives=num_negatives
                ).to(device)

                run_name = (
                    f"reg_{lambda_reg}_nce_{lambda_nce}"
                    f"_tau_{temperature}_neg{num_negatives}"
                )
                run_dir  = os.path.join(checkpoint_dir, run_name)
                os.makedirs(run_dir, exist_ok=True)

                # (Re)initialize
                generator     = ResNetGenerator().to(device)
                discriminator = PatchDiscriminator().to(device)
                stn           = STN().to(device)
                encoder       = PatchEncoder(in_ch=3, feat_dim=256).to(device)

                init_weights(generator,     "normal", 0.02)
                init_weights(discriminator, "normal", 0.02)
                init_weights(stn,           "normal", 0.02)
                init_weights(encoder,       "normal", 0.02)

                # Reset STN to identity
                with torch.no_grad():
                    stn.fc_loc[2].weight.zero_()
                    stn.fc_loc[2].bias.copy_(torch.tensor([1.,0.,0.,0.,1.,0.], device=device))

                # Optimizers & schedulers
                optimizer_G = torch.optim.Adam(
                    list(generator.parameters()) +
                    list(stn.parameters())       +
                    list(encoder.parameters()),
                    lr=2e-4, betas=(0.5,0.999)
                )
                optimizer_D = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=2e-4, betas=(0.5,0.999)
                )
                sched_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
                sched_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

                print(f"\n=== Starting run {run_name} ===")
                for epoch in range(epochs):
                    for i, (he, ihc) in enumerate(dataloader):
                        he, ihc = he.to(device), ihc.to(device)

                        ###### Discriminator step ######
                        # real
                        d_real = discriminator(he, ihc)
                        d_real_loss = F.mse_loss(d_real, torch.ones_like(d_real))
                        # fake
                        fake_ihc = generator(he).detach()
                        d_fake = discriminator(he, fake_ihc)
                        d_fake_loss = F.mse_loss(d_fake, torch.zeros_like(d_fake))
                        # total
                        d_loss = 0.5 * (d_real_loss + d_fake_loss)

                        optimizer_D.zero_grad()
                        d_loss.backward()
                        optimizer_D.step()

                        ###### Generator + STN + Encoder step ######
                        fake_ihc     = generator(he)
                        warped_ihc, theta = stn(he, ihc)

                        # build valid‐mask for warped regions
                        grid = F.affine_grid(theta, ihc.size(), align_corners=True)
                        valid_mask = (
                            (grid[...,0] >= -1) & (grid[...,0] <= 1) &
                            (grid[...,1] >= -1) & (grid[...,1] <= 1)
                        ).float().unsqueeze(1).expand(-1,3,-1,-1)

                        # 1) adversarial
                        d_out = discriminator(he, fake_ihc)
                        g_adv = F.mse_loss(d_out, torch.ones_like(d_out))

                        # 2) pixel L1
                        g_pix = F.l1_loss(fake_ihc * valid_mask,
                                        warped_ihc * valid_mask)

                        # 3) multi‐scale L1 on‐the‐fly
                        fake_pyr = gaussian_pyramid(fake_ihc, levels=3)[1:]
                        real_pyr = gaussian_pyramid(warped_ihc, levels=3)[1:]
                        g_multi = 0
                        for f, r in zip(fake_pyr, real_pyr):
                            mask_lvl = F.interpolate(valid_mask, size=f.shape[2:], mode='nearest')
                            g_multi += (
                                F.l1_loss(f * mask_lvl, r * mask_lvl, reduction='sum')
                                / (mask_lvl.sum() + 1e-6)
                            )

                        # 4) STN reg
                        id_theta = (
                            torch.tensor([1.,0.,0.,0.,1.,0.], device=device)
                                .view(1,2,3)
                                .expand(theta.size(0), -1, -1)
                        )
                        reg_loss = F.mse_loss(theta, id_theta)

                        # 5) PatchNCE
                        feat_q = encoder(fake_ihc)
                        feat_k = encoder(warped_ihc)

                        g_nce  = patchnce_loss(feat_q, feat_k)

                        # 6) total G loss
                        g_loss = (
                            g_adv
                            + 100.0 * g_pix
                            + g_multi
                            + lambda_reg * reg_loss
                            + lambda_nce * g_nce
                        )

                        optimizer_G.zero_grad()
                        g_loss.backward()
                        optimizer_G.step()

                        # log every 10 iters
                        if i % 10 == 0:
                            print(
                                f"[{run_name}] Epoch {epoch+1}/{epochs} | "
                                f"Batch {i}/{len(dataloader)} | "
                                f"D loss  : {d_loss.item():.4f} | "
                                f"G loss  : {g_loss.item():.4f} | "
                                f"NCE loss: {g_nce.item():.4f}"
                            )

                    sched_G.step()
                    sched_D.step()

                    # checkpoint per epoch
                    torch.save({
                        'epoch': epoch,
                        'generator':     generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'stn':           stn.state_dict(),
                        'encoder':       encoder.state_dict(),
                        'optim_G':       optimizer_G.state_dict(),
                        'optim_D':       optimizer_D.state_dict(),
                    }, os.path.join(run_dir, f"ckpt_epoch_{epoch+1}.pth"))

                print(f"=== Finished run {run_name} ===\n")

                visualize_samples(generator, stn, dataloader, num_samples=3)
                if (epoch + 1) % 5 == 0:
                    save_checkpoint(
                        epoch + 1, generator, discriminator,
                        optimizer_G, optimizer_D,
                        g_loss.item(), d_loss.item()
                    )

