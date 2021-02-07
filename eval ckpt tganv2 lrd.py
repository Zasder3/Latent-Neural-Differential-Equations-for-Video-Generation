import torch
import numpy as np
from models.tganv2_gen import Generator_CLSTM
from evaluation_metrics import calculate_inception_score_confidence

path = 'UCF101/tganv2_lrd'

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

    state_dicts = torch.load(f'checkpoints/{path}/state_normal59000.ckpt')

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    gen.cpu()
    results = calculate_inception_score_confidence(gen)
    np.save('results.npy', results)
    print(results)
