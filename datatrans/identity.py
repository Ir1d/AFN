import torch.nn as nn
from torchvision import transforms


def get():
    return transforms.Compose([transforms.ToTensor()])
