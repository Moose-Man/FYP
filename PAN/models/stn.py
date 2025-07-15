import torch, torch.nn as nn, torch.nn.functional as F

class STN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_net = nn.Sequential(                 # 6-→32-→64 conv tower
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 6)
        )
        # start precisely at identity
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0, 0,1,0], dtype=torch.float))

    def forward(self, A, B):
        x     = torch.cat([A, B], 1)               # (N,6,H,W)
        theta = self.fc_loc(self.loc_net(x).view(x.size(0), -1)).view(-1,2,3)
        grid  = F.affine_grid(theta, B.size(), align_corners=True)
        B_w   = F.grid_sample(B, grid, align_corners=True)      # warped target
        return B_w, theta, grid
