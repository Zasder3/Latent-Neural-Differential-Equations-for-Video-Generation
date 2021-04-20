import torch
import numpy as np
from models.tgan_ode import VideoGenerator
from evaluation_metrics import calculate_inception_score_confidence

path = 'ucf101/tgan_svc_ode_lin'

if __name__ == '__main__':
    # gen model
    gen = VideoGenerator(linear=True)

    state_dicts = torch.load(f'checkpoints/{path}/state_normal97000.ckpt')

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    gen.cpu()
    results = calculate_inception_score_confidence(gen, test=False, zdim=100)
    np.save('results/TGAN ODE Linear.npy', results)
    print(results)
