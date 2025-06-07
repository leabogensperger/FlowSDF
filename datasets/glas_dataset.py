import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets.transform_factory import transform_factory


class GlaSDataset(Dataset):
    def __init__(self, data_path, csv_file, cfg, batch_size, test_mode=False):
        self.data_path = data_path
        self.csv_file = csv_file
        self.data = pd.read_csv(self.csv_file)

        self.transform = None
        self.inv_normalize = None

        self.corr_mode = cfg.corr_mode
        self.img_cond = cfg.img_cond
        self.sz = cfg.sz

        self.batch_size = batch_size
        self.test_mode = test_mode

    def __len__(self):
        if self.test_mode:
            return len(self.data)
        return max(len(self.data), self.batch_size)

    def __getitem__(self, idx):
        idx = idx % len(self.data)

        img_path = self.data_path + self.data.loc[idx]['filename']
        mask_path = self.data_path + self.data.loc[idx]['maskname']
        mask_ls_path = self.data_path + self.data.loc[idx]['maskdtname']

        if 'rgb' in self.data_path:
            img = cv2.imread(img_path).astype(np.float32) / 255.
        else:
            img = cv2.imread(img_path, 0).astype(np.float32) / 255.

        mask = np.load(mask_ls_path).astype(np.float32)

        transform_cfg = {
            'hflip': np.random.rand(),
            'vflip': np.random.rand(),
            'corr_type': 1,
            'img_cond': self.img_cond,
        }

        ret = {
            'image': img,
            'mask': mask,
            'name': str(img_path.split('/')[-1][:-4])
        }

        self.transform(transform_cfg)(ret)
        return ret
