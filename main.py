"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
from torch import autograd
from loguru import logger
from tqdm import tqdm
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from trainer import NTIRE20Model
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TensorBoardLogger
from time import strftime, gmtime
from IPython import embed
import os
import torch
torch.backends.cudnn.benchmark = True


def main(hparams):
    logger.remove()
    os.makedirs(f'./logs/{hparams.expname}', exist_ok=True)
    log_file = f'./logs/{hparams.expname}/{hparams.timestamp}.txt'
    logger.add(lambda msg: tqdm.write(msg, end=""),
               format="{time:HH:mm:ss} {message}")
    logger.add(log_file, rotation="20 MB", backtrace=True, diagnose=True)
    # init module
    # Load previous params
    # embed()
    model = NTIRE20Model(hparams)
    if (hparams.checkpoint != ""):
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f'Trigger whole load from {hparams.checkpoint}')
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            logger.warning(f'Strict Loading resulted error')
            logger.warning(f'Retry loading with strict=False')
            model.load_state_dict(
                checkpoint['state_dict'], strict=False)
        logger.info(f'Loading weights from {hparams.checkpoint}')
    if (hparams.mode == 'train'):
        checkpoint_callback = ModelCheckpoint(
            filepath=f"{os.getcwd()}/weights/{hparams.expname}/{hparams.timestamp}/",
            monitor='avg_psnr',
            mode='max',
        )
        early_stopping = EarlyStopping(
            monitor='avg_psnr',
            mode='max',
            patience=10,
        )

        tblogger = TensorBoardLogger(
            "tb_logs", name=hparams.expname, version=hparams.timestamp)

        # most basic trainer, uses good defaults
        trainer = Trainer(
            max_epochs=hparams.max_epochs,
            min_epochs=30,
            gpus=hparams.gpus,
            nb_gpu_nodes=hparams.nodes,
            distributed_backend='dp',
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stopping,
            logger=tblogger,
            val_percent_check=1,
            test_percent_check=1,
            num_sanity_val_steps=1,
        )
        # with autograd.detect_anomaly():
        trainer.fit(model)
    elif (hparams.mode == 'test'):
        trainer = Trainer(
            gpus=hparams.gpus,
            nb_gpu_nodes=hparams.nodes,
            distributed_backend='dp',
        )
        os.makedirs(f'./res/{hparams.timestamp}', exist_ok=True)
        if hparams.self_ensemble:
            logger.info("Self ensembling enabled")
        else:
            logger.info("Self ensembling disabled")
        trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--self-ensemble', action='store_true',
                        help='enable naive test time augmentation')
    parser.add_argument('--useAug', action='store_true')
    parser.add_argument('--gpus', type=str, default='-1')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('-E', '--expname', type=str,
                        default='', help='indicate experiment name')
    parser.add_argument('-M', '--modelname', type=str,
                        required=True, help='indicate model module')
    parser.add_argument('-L', '--lossname', type=str,
                        default='naive', help='indicate loss module')
    parser.add_argument('-T', '--inputtrans', type=str, default='identity',
                        help='indicate data trans module for input, data trans take effect before aug')
    parser.add_argument('-t', '--gttrans', type=str, default='identity',
                        help='indicate data trans module for gt, data trans take effect before aug')
    parser.add_argument('--timestamp', type=str,
                        default=strftime("%m-%d_%H-%M-%S", gmtime()), help='exp timestamp')
    parser.add_argument('-C', '--checkpoint', type=str,
                        default='', help='weights file location')
    parser.add_argument('--seed', type=int, default=998244353,
                        help='random seed, set -1 if you don\'t want a manual seed')
    parser.add_argument('--cyh', action='store_true',
                        help='resolve trainning interface conflict')
    parser.add_argument('--fullLoad', action='store_true',
                        help='load full model')
    parser.add_argument('--loadOptim', action='store_true',
                        help='load full model')
    # specify the timestamp to load previous models

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = NTIRE20Model.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    if hparams.expname == '':
        hparams.expname = hparams.modelname

    main(hparams)
    # try:
    #     main(hparams)
    # except:
    #     embed()
