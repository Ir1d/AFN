import os
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
import torchvision as tv
from data import DemoireSingleDataset, TestSet
from losses.import_helper import get_loss
from kornia.losses import psnr_loss
from kornia.utils import tensor_to_image
from loguru import logger
from tqdm import tqdm
from PIL import Image
from IPython import embed
from models.import_helper import get_model
from datatrans.import_helper import get_trans
import numpy as np
import os
import pickle as pkl
from torch.optim.lr_scheduler import CosineAnnealingLR
from CyclicLR import CyclicCosAnnealingLR


class NTIRE20Model(pl.LightningModule):

    def __init__(self, hparams):
        # logger
        logger.info(str(hparams))
        # res_folder
        self.res_folder = f'./res/{hparams.timestamp}'

        if hparams.seed != -1:
            logger.info(f"Using manual seed {hparams.seed}")
            random.seed(hparams.seed)
            torch.manual_seed(hparams.seed)
            torch.cuda.manual_seed_all(hparams.seed)
            np.random.seed(hparams.seed)
        else:
            logger.info("Using system determined RNG state")

        super(NTIRE20Model, self).__init__()

        self.hparams = hparams
        self.model = get_model(f'./models/{hparams.modelname}.py')
        # embed()
        self.loss = get_loss(f'./losses/{hparams.lossname}.py')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        data = batch
        y_hat = self.forward(data)
        loss = self.loss(y_hat, data)
        if (batch_idx % self.hparams.log_interval == 0):
            # w = torch.cat((data['input'][0], y_hat['image'][0], data['gt'][0]), dim=2)
            # self.logger.experiment.add_image(f'train-{self.current_epoch}', w, global_step=self.global_step)
            # Loss

            loss_msg = f"Epoch[{self.current_epoch}]:"
            for k, v in loss.items():
                self.logger.experiment.add_scalar(
                    f"train{k}", v, global_step=self.global_step)
                loss_msg = loss_msg + f" {k} {v :.4f}"
            logger.info(loss_msg)

            # print weight distribution
            # for i, (name, net_param) in enumerate(self.model.last_layer.named_parameters()):
            #     if name[:7] == 'module.':
            #         name = name[7:]
            #     name = name.replace('.', '/', 1)
            #     if 'relu' in name:
            #         self.logger.experiment.add_histogram(f'relu_{name}', net_param, global_step=self.global_step)
            #     elif 'bn' in name:
            #         self.logger.experiment.add_histogram(f'bn_{name}', net_param, global_step=self.global_step)
            #     else:
            #         self.logger.experiment.add_histogram(name, net_param, global_step=self.global_step)
        return {'loss': loss['tot']}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch
        y_hat = self.forward(data)
        y_hat['image'] = torch.clamp(y_hat['image'], min=0, max=1)
        if (batch_idx % self.hparams.log_interval == 0):
            pass
            # w = torch.cat((data['input'][0], y_hat['image'][0], data['gt'][0]), dim=2)
            # self.logger.experiment.add_image(f'val-{self.current_epoch}', w, global_step=self.global_step)
        if (self.hparams.cyh or (y_hat['image'].shape[1] == 3 and not 'cwep_nobn' in self.hparams.modelname)):
            gt = data['gt']
        else:
            gt = data['edge_gt']
        return {'psnr': psnr_loss((y_hat['image'] * 255).int().float(), (gt * 255).int().float(), 255), 'mse': F.mse_loss(y_hat['image'], gt)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        avg_mse = torch.stack([x['mse'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            f"valPSNR", avg_psnr, global_step=self.current_epoch)
        self.logger.experiment.add_scalar(
            f"valMSE", avg_mse, global_step=self.current_epoch)
        logger.info(
            f"Epoch[{self.current_epoch}]: PSNR {avg_psnr}, MSE {avg_mse :.4f}")
        return {'avg_psnr': avg_psnr, 'avg_mse': avg_mse}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data = batch

        with torch.no_grad():
            resus = list()
            resus.append(self.forward(data)['image'])

            final_resu = torch.cat(resus, dim=0).clamp(
                min=0, max=1).cpu().numpy()

        img_ = np.mean(final_resu, axis=0)

        if (img_.shape[0] == 3):
            img = Image.fromarray(
                ((img_.transpose((1, 2, 0))) * 255).astype(np.uint8))
            filename = os.path.join(self.res_folder, batch['name'][0])
            file_folder = os.path.dirname(filename)
            os.makedirs(file_folder, exist_ok=True)
            logger.info(filename)
            img.save(filename)

        return {}

    def test_end(self, outputs):
        return {}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.hparams.learning_rate)
        if self.hparams.cyclicLR:
            logger.info('Enable Cyclic LR')
            scheduler = CyclicCosAnnealingLR(opt, milestones=[
                                             50, 100, 150, 200], decay_milestones=[50, 100, 150, 200], eta_min=1e-6)
        else:
            scheduler = CosineAnnealingLR(
                opt, eta_min=1e-6, T_max=self.hparams.max_epochs)
        return [opt], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(DemoireSingleDataset(split='train', useAug=self.hparams.useAug, inputtrans=f'./datatrans/{self.hparams.inputtrans}.py', gttrans=f'./datatrans/{self.hparams.gttrans}.py'), batch_size=self.hparams.batch_size, num_workers=32, shuffle=True, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        ValSet = DemoireSingleDataset(
            split='val', inputtrans=f'./datatrans/{self.hparams.inputtrans}.py', gttrans=f'./datatrans/{self.hparams.gttrans}.py')
        return DataLoader(ValSet, batch_size=1, num_workers=4)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(TestSet(inputtrans=f'./datatrans/{self.hparams.inputtrans}.py'), batch_size=1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--log_interval', default=100, type=int)

        # training specific (for this model)
        parser.add_argument('--max_epochs', default=200, type=int)
        parser.add_argument(
            '--cyclicLR', action='store_true', help='load full model')

        return parser
