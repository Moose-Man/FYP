import torch, torch.nn as nn, torch.nn.functional as F

class STN(nn.Module):
    def __init__(self):
        super().__init__()
        # match your pyramid_Pix2Pix_ver8 design
        self.loc_net = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 6)
        )
        # init to identity
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0],dtype=torch.float))

    def forward(self, A, B):
        x = torch.cat([A, B], dim=1)
        xs = self.loc_net(x).view(x.size(0), -1)
        theta = self.fc_loc(xs).view(-1,2,3)
        grid  = F.affine_grid(theta, B.size(), align_corners=True)
        B_warp = F.grid_sample(B, grid, align_corners=True)
        return B_warp, theta, grid

