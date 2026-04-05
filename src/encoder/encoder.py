import torch.nn as nn
from .e_blocks import ConvBlock

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        self.e1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)

        self.e3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.e4 = ConvBlock(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        self.e5 = ConvBlock(base_ch*8, base_ch*16)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        e4 = self.e4(self.pool3(e3))
        e5 = self.e5(self.pool4(e4))

        return e1, e2, e3, e4, e5
