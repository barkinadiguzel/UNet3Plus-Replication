import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(pred.shape[0], -1)
        target_flat = target.contiguous().view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(1)
        dice = (2. * intersection + self.smooth) / \
               (pred_flat.sum(1) + target_flat.sum(1) + self.smooth)

        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        return bce_loss + dice_loss
