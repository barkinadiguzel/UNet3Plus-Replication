import torch
import torch.nn as nn
import torch.nn.functional as F

class FullScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(sum(in_channels_list), out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def resize(self, x, target_size):
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

    def forward(self, features, target_size):
        resized = [self.resize(f, target_size) for f in features]
        x = torch.cat(resized, dim=1)
        return self.conv(x)
