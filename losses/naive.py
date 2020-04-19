import torch.nn as nn
import torch
import torch.nn.functional as F
from kornia.losses import SSIM


def color_loss(output, gt):
    img_ref = F.normalize(output, p=2, dim=1)
    ref_p = F.normalize(gt, p=2, dim=1)
    loss_cos = 1 - torch.mean(F.cosine_similarity(img_ref, ref_p, dim=1))
    # loss_cos = self.mse(img_ref, ref_p)
    return loss_cos


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.l1 = nn.L1Loss()
        # self.ssim = SSIM(5, reduction='mean')
        # kornia 里面的定义是 loss = (1 - SSIM) / 2

    def forward(self, out, gt):
        l1 = self.l1(out['image'], gt['gt'])
        tot = l1
        return {'tot': tot, 'L1': l1}
        # return {'tot': tot, 'L1': l1, 'SSIM': ssim}


def get():
    return MyLoss()
