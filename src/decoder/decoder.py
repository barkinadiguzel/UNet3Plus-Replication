import torch.nn as nn
from .d_blocks import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()

        self.d4 = DecoderBlock([base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16], base_ch*4)
        self.d3 = DecoderBlock([base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*4], base_ch*2)
        self.d2 = DecoderBlock([base_ch, base_ch*2, base_ch*4, base_ch*2, base_ch*4], base_ch)
        self.d1 = DecoderBlock([base_ch, base_ch*2, base_ch, base_ch*2, base_ch*4], base_ch)

    def forward(self, e1, e2, e3, e4, e5):
        d4 = self.d4([e1, e2, e3, e4, e5], e4.shape[2:])
        d3 = self.d3([e1, e2, e3, e4, d4], e3.shape[2:])
        d2 = self.d2([e1, e2, e3, d3, d4], e2.shape[2:])
        d1 = self.d1([e1, e2, d2, d3, d4], e1.shape[2:])

        return d1, d2, d3, d4
