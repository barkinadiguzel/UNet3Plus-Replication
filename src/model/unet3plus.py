import torch.nn as nn
from ..encoder.encoder import Encoder
from ..decoder.decoder import Decoder
from ..supervision.deep_supervision import DeepSupervision

class UNet3Plus(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, base_ch=64):
        super().__init__()

        self.encoder = Encoder(in_ch, base_ch)
        self.decoder = Decoder(base_ch)

        self.deep_sup = DeepSupervision(
            [base_ch, base_ch, base_ch*2, base_ch*4],
            num_classes
        )

    def forward(self, x):

        e1, e2, e3, e4, e5 = self.encoder(x)

        d1, d2, d3, d4 = self.decoder(e1, e2, e3, e4, e5)

        outputs = self.deep_sup([d1, d2, d3, d4])

        return outputs
