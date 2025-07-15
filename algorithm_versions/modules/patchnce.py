# models/patchnce.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion   = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        """
        feat_q, feat_k: [B, C, H*W] flattened features from two views
        returns scalar loss
        """
        B, C, HW = feat_q.size()
        feat_q = feat_q / feat_q.norm(dim=1, keepdim=True)
        feat_k = feat_k / feat_k.norm(dim=1, keepdim=True)

        # positive logits: [B, HW]
        l_pos = torch.bmm(feat_q.transpose(1,2), feat_k).view(-1, 1)

        # negative logits: [B, HW, B*HW]
        l_neg = torch.mm(feat_q.transpose(0,2).reshape(-1, C),
                         feat_k.reshape(C, -1)).detach()
        mask = torch.eye(l_neg.size(0), device=feat_q.device).bool()
        l_neg = l_neg.masked_fill_(mask, -10.0)  # hide positives
        l_neg = l_neg.view(l_pos.size(0), -1)

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=feat_q.device)
        return self.criterion(logits, labels)

###############################################################################
# No-op replacement — same call-signature, returns 0 so the loop is oblivious #
###############################################################################
class IdentityPatchNCELoss(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0, device=kwargs.get('device', 'cpu'))
