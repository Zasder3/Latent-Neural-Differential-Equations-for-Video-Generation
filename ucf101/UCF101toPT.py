import os
import torch
import torchvision.io as io
from tqdm import tqdm

for f in tqdm(os.listdir('D:/Video Datasets/UCF101_64px')):
    i, _, _ = io.read_video('D:/Video Datasets/UCF101_64px/{}'.format(f))
    torch.save(i, 'D:/Video Datasets/UCF101_torch/{}.pt'.format(f[:-4]))
