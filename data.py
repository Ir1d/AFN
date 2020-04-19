import glob
import pickle as pkl

import albumentations as A
import numpy as np
import os
from PIL import Image
from albumentations import RandomRotate90, Flip, Transpose, RandomResizedCrop
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
from datatrans.import_helper import get_trans
TR_INPUT = '/data/ntire/demoire_single/train/input/'
TR_GT = '/data/ntire/demoire_single/train/gt/'


class DemoireSingleDataset(Dataset):
    """Demoire Single dataset."""

    def __init__(self, split='train', train_ratio=0.95, useAug=True, inputtrans="", gttrans="", fast_load=True):
        """
        train_ratio: ratio of train
        1e4 pairs in all
        """
        assert inputtrans
        assert gttrans
        # self.input_dir = "/data/ntire/rawdata/train/input/"
        self.input_dir = TR_INPUT
        # ex: 000006_3.png
        # self.gt_dir = "/data/ntire/rawdata/train/gt/"
        self.gt_dir = TR_GT
        # ex: 000003_gt.png
        self.input_list = glob.glob(self.input_dir + '/*.png')
        self.gt_list = glob.glob(self.gt_dir + '/*.png')
        self.input_transform = get_trans(inputtrans)
        self.gt_transform = get_trans(gttrans)
        self.input_list = np.array(sorted(self.input_list))
        self.gt_list = np.array(sorted(self.gt_list))

        self.split = split
        self.len = len(self.input_list)
        self.useAug = useAug
        if (self.split == 'train'):
            with open('train.pkl', 'rb') as ff:
                train_ids = pkl.load(ff)
            self.input_list = self.input_list[train_ids]
            self.gt_list = self.gt_list[train_ids]
            # aug
            if (useAug):
                self.aug = A.Compose([RandomRotate90(p=0.1), Flip(p=0.1), Transpose(p=0.1),
                                      RandomResizedCrop(height=128, width=128, p=0.1)],
                                     additional_targets={'image': 'image', 'gt': 'image'})
        elif (self.split == 'val'):
            with open('val.pkl', 'rb') as ff:
                val_ids = pkl.load(ff)
            self.input_list = self.input_list[val_ids]
            self.gt_list = self.gt_list[val_ids]
        self.fast_load = fast_load
        if self.fast_load:
            self.input_images = {}
            idx = [i for i in range(len(self.input_list))]
            target = ['self.input_images'] * len(self.input_list)
            inp = zip(idx, self.input_list, target)
            from tqdm import tqdm
            for i in tqdm(list(inp), desc='loading input images'):
                self.save_in_memory(i)
            # Pool didn't work ????
            # with Pool(5) as p:
            #     p.map(self.save_in_memory, list(inp))

            self.gt_images = {}
            idx = [i for i in range(len(self.gt_list))]
            target = ['self.gt_images'] * len(self.gt_list)
            inp = zip(idx, self.gt_list, target)
            for i in tqdm(list(inp), desc='loading gt images'):
                self.save_in_memory(i)
            # with Pool(5) as p:
            # 	p.map(self.save_in_memory, list(inp))

        self.len = len(self.input_list)
        self.totensor = ToTensor()

        logger.info(f'Loaded Dataset split:{split}, len:{self.len}')

    def save_in_memory(self, inp):
        idx, path, target = inp
        exec(f'{target}[{idx}] = self.load_image("{path}")')

    def load_image(self, load_path):
        x = np.array(Image.open(load_path))
        return x

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # idx = idx % self.len
        input_name = self.input_list[idx]
        gt_name = self.gt_list[idx]

        if not self.fast_load:
            x_input = self.load_image(input_name)
            x_gt = self.load_image(gt_name)
        else:
            x_input = self.input_images[idx]
            x_gt = self.gt_images[idx]

        if self.useAug and self.split == 'train':
            transformed = self.aug(image=x_input, gt=x_gt)
            x_input, x_gt = transformed['image'], transformed['gt']

        r_input = self.totensor(x_input)
        r_gt = self.totensor(x_gt)
        input = self.input_transform(x_input)
        gt = self.gt_transform(x_gt)

        data = {'input': input, 'gt': gt,
                'name': os.path.basename(input_name)}
        return data


class TestSet(Dataset):
    def __init__(self, inputtrans="", pt=False):
        assert inputtrans
        self.input_dir = '/data/ntire/demoire_single/val/'
        # self.input_dir = '/data/ntire/rawdata/val_input'
        self.input_list = glob.glob(self.input_dir + '/*.png')
        self.input_transform = get_trans(inputtrans)
        self.len = len(self.input_list)
        self.pt = pt
        if (self.pt):
            self.edge_list = 'res/cwep_xy/'
        self.totensor = ToTensor()
        logger.info(f'Loaded TestSet, len:{self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # idx = idx % self.len
        input_name = self.input_list[idx]

        img = Image.open(input_name)
        r_input = self.totensor(img)
        input = self.input_transform(img)
        filename = os.path.basename(input_name)[0]
        data = {'input': input, 'name': filename}

        return data
