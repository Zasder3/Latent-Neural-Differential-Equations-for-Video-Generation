import torch
import os
import numpy as np
from models.tganv2_gen import Generator_CLSTM
from evaluation_metrics import calculate_inception_score
from tqdm.gui import tqdm

path = 'UCF101/tganv2'

if __name__ == '__main__':
    # gen model
    gen = Generator_CLSTM(
        tempc=512,
        zt_dim=4,
        upchannels=[256],
        subchannels=[128, 64, 32],
        n_frames=16,
        colors=3
    ).cuda()

    scores = []
    iters = []
    best = 0
    best_name = None
    for name in tqdm(os.listdir(f'D:/model_code/checkpoints/{path}/')):
        state_dicts = torch.load(f'D:/model_code/checkpoints/{path}/{name}')

        gen.load_state_dict(state_dicts['model_state_dict'][0])
        gen.cpu()
        scores.append(calculate_inception_score(gen))
        iters.append(name[12:-5])
        if scores[-1] > best:
            best = scores[-1]
            best_name = name
        print(iters[-1], scores[-1])
    print(best_name)
    np.save('tganv2_inception.npy', [iters, scores])
