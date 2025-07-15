# models/patchnce.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

###############################################################################
# Original PatchNCE loss — *identical* to the one already sitting in ver11.py #
###############################################################################

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
    """
    Contrastive loss used in CUT / Pix2PixHD-NCE.
    For each query patch, we keep its positive pair and
    *randomly sample `num_negatives` other patches* (across batch & spatial dims)
    to serve as negatives.

    Args
    ----
    temperature : float
    num_negatives : int  (default 128, as in ver 11)
    """
    def __init__(self, temperature=0.07, num_negatives=128):
        super().__init__()
        self.temperature    = temperature
        self.num_negatives  = num_negatives
        self.criterion      = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        """
        Accepts either
        • [B, C, H, W] feature maps  **or**
        • [B, C, H*W] flattened maps
        and computes PatchNCE with `self.num_negatives` negatives.
        """
        if feat_q.dim() == 4:                         # [B, C, H, W] → flatten
            B, C, H, W = feat_q.shape
            HW = H * W
            feat_q = feat_q.reshape(B, C, HW)
            feat_k = feat_k.reshape(B, C, HW)
        else:                                         # already flat
            B, C, HW = feat_q.shape

        N_total  = B * HW                             # total patches in batch

        # L2-normalise
        feat_q = F.normalize(feat_q, dim=1)           # [B, C, HW]
        feat_k = F.normalize(feat_k, dim=1)

        # reshape to [N_total, C]
        feat_q = feat_q.permute(0, 2, 1).reshape(-1, C)
        feat_k = feat_k.permute(0, 2, 1).reshape(-1, C)

        # Positive logits: dot-product with own positive key
        l_pos = (feat_q * feat_k).sum(dim=1, keepdim=True)  # [N_total, 1]

        # Build negative pool (all keys)
        with torch.no_grad():
            # indices of all patches except self
            all_idx = torch.arange(N_total, device=feat_q.device)
            neg_indices = []
            for i in range(N_total):
                choices = torch.cat((all_idx[:i], all_idx[i+1:]))
                # random.sample is on cpu; use torch.randint for speed
                rand_ids = torch.randint(
                    0, choices.size(0),
                    (self.num_negatives,),
                    device=feat_q.device)
                neg_indices.append(choices[rand_ids])
            neg_indices = torch.stack(neg_indices)           # [N_total, num_negatives]

        # Gather negative keys and compute logits
        feat_k_exp = feat_k[neg_indices]                     # [N_total, num_neg, C]
        l_neg = torch.bmm(
            feat_q.unsqueeze(1),                             # [N_total, 1, C]
            feat_k_exp.permute(0, 2, 1)                      # [N_total, C, num_neg]
        ).squeeze(1)                                         # [N_total, num_neg]

        # Concatenate – positives first
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(N_total, dtype=torch.long, device=feat_q.device)
        return self.criterion(logits, labels)

# ─── No-op replacement when PatchNCE is disabled ───────────────────────────────
class IdentityPatchNCELoss(nn.Module):
    def forward(self, *_, **__):
        # return scalar 0 on correct device
        for arg in _:
            if torch.is_tensor(arg):
                return torch.tensor(0.0, device=arg.device)
        return torch.tensor(0.0)