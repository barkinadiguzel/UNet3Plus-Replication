import torch
import torch.nn as nn
import torch.nn.functional as F
from ..aggregation.full_scale_fusion import FullScaleFusion

class DecoderBlock(nn.Module):
    def __init__(self, in_channels_list, out_ch):
        super().__init__()
        self.fuse = FullScaleFusion(in_channels_list, out_ch)

    def forward(self, features, target_size):
        return self.fuse(features, target_size)
