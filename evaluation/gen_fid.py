import torch
import cupy as cp
import numpy as np
import chainer
import cv2
from ucf101.UCF101DatasetTGAN import UCF101
from evaluation.c3d_ft import C3DVersion1
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":
    Path('temp/').mkdir(parents=True, exist_ok=True)
    conf = '../train.json'
    dset = '../train.h5'

    test = UCF101(dset, conf, img_size=192)
    loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False,
                                         drop_last=False)
    # load calc model
    dir = 'evaluation/pfnet/chainer/models/conv3d_deepnetA_ucf.npz'
    model = C3DVersion1(pretrained_model=dir).to_gpu()
    mean = np.load('evaluation/mean2.npz')['mean'].astype('f')
    mean = mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]

    print("Creating video files")
    print("Calculating probabilities")
    iters = 0
    out = []
    for j in tqdm(range(10)):
        for i, x in enumerate(loader):
            iters += 1
            x = x.numpy().transpose(0, 2, 1, 3, 4)
            n, c, f, h, w = x.shape
            x = x.transpose(0, 2, 3, 4, 1).reshape(n * f, h, w, c)
            x = x * 128 + 128
            x_ = np.zeros((n * f, 128, 128, 3))
            for t in range(n * f):
                x_[t] = np.asarray(
                    cv2.resize(x[t], (128, 128), interpolation=cv2.INTER_CUBIC))
            x = x_.transpose(3, 0, 1, 2).reshape(3, n, f, 128, 128)
            s = x.shape
            assert x.shape == s
            x = x[::-1] - mean  # mean file is BGR while model outputs RGB
            x = x[:, :, :, 8:8 + 112, 8:8 + 112].astype('f')
            x = x.transpose(1, 0, 2, 3, 4)
            x = cp.asarray(x)
            with chainer.using_config('train', False) and \
                 chainer.no_backprop_mode():
                out.append(model(x, layers=['fc7'])['fc7'].data.get())

    del x

    out = np.concatenate(out, axis=0)
    mean = np.mean(out, axis=0)
    cov = np.cov(out.T)
    np.save('fid_mean.npy', mean)
    np.save('fid_cov.npy', cov)
