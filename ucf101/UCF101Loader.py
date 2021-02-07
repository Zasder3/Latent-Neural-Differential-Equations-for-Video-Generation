import os
import random
import torch
import torchvision
from torch.utils import data



class UCF101(data.Dataset):
    def __init__(self, n_frames=16):
        self.ucf_path = 'D:/Video Datasets/UCF101_torch/'
        self.videos = os.listdir(self.ucf_path)
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        X = torch.load(self.ucf_path + self.videos[index]).float()
        c = random.randint(0, X.shape[0]-16)
        X = X[c:c+16]
        X = X.permute(0, 3, 1, 2) / 255 * 2 -1
        return X, torch.zeros(1)