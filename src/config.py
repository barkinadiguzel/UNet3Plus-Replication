import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_CHANNELS = 3
NUM_CLASSES = 1

BASE_CHANNEL = 64
