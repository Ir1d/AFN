import torch.nn as nn
from kornia.losses import SSIM


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.l1 = nn.L1Loss()
        # kornia 里面的定义是 loss = (1 - SSIM) / 2

    def forward(self, out, gt):
        l1 = self.l1(out['image'], gt['gt'])
        return {'tot': l1, 'L1': l1}


def get():
    return MyLoss()
