# models/stn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    """Real STN exactly as in ver11, no change."""
    def __init__(self):
        super().__init__()
        self.loc_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),     nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 6)
        )
        # identity init
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1,0,0, 0,1,0], dtype=torch.float)
        )

    def forward(self, he, ihc):
        x      = torch.cat([he, ihc], dim=1)
        theta  = self.fc_loc(self.loc_net(x).view(x.size(0), -1)).view(-1, 2, 3)
        grid   = F.affine_grid(theta, he.size(), align_corners=True)
        warped = F.grid_sample(ihc, grid, align_corners=True)
        return warped, theta


class IdentitySTN(nn.Module):
    """
    Drop-in replacement when the user **disables** STN.
    Makes the training loop oblivious to the toggle:
    - returns the original IHC unchanged
    - theta = identity (required shape, so later code keeps working)
    """
    def forward(self, he, ihc):
        N = ihc.size(0)
        device = ihc.device
        theta = torch.tensor([[1,0,0],[0,1,0]], dtype=ihc.dtype,
                             device=device).repeat(N,1,1)
        return ihc, theta
