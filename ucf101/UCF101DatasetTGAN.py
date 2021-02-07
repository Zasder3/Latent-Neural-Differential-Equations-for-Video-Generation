import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils import data


class UCF101(data.Dataset):
    def __init__(self, h5path, config_path):
        self.h5file = h5py.File(h5path, 'r')
        self.dset = self.h5file['image']
        self.conf = pd.read_pickle(config_path)
        self.ind = self.conf.index.tolist()
        self.n_frames = 16
        self.img_size = 64

    def __len__(self):
        return len(self.conf)

    def _crop_center(self, x):
        x = x[:, :, :, 10:10 + self.img_size]
        assert x.shape[2] == self.img_size
        assert x.shape[3] == self.img_size
        return x

    def __getitem__(self, i):
        mov_info = self.conf.loc[self.ind[i]]
        length = mov_info.end - mov_info.start
        offset = np.random.randint(length - self.n_frames) \
            if length > self.n_frames else 0
        x = self.dset[mov_info.start + offset:
                      mov_info.start + offset + self.n_frames]
        x = self._crop_center(x)
        return torch.tensor((x - 128.0) / 128.0, dtype=torch.float)
