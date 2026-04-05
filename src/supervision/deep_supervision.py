import torch.nn as nn

class DeepSupervision(nn.Module):
    def __init__(self, in_channels_list, num_classes):
        super().__init__()

        self.heads = nn.ModuleList([
            nn.Conv2d(ch, num_classes, 1) for ch in in_channels_list
        ])

    def forward(self, features):
        return [head(f) for head, f in zip(self.heads, features)]
