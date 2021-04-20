import torch
import numpy as np
from models.tgan import VideoGenerator
from evaluation_metrics import calculate_inception_score_confidence

path = 'ucf101/tgan_svc'

if __name__ == '__main__':
    # gen model
    gen = VideoGenerator()

    state_dicts = torch.load(f'checkpoints/{path}/state_normal90000.ckpt')

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    gen.cpu()
    results = calculate_inception_score_confidence(gen, test=False, zdim=100)
    np.save('results/TGAN SVC.npy', results)
    print(results)
