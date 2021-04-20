import torch
import numpy as np
from models.mocogan_ode import VideoGenerator, ODEFuncDeep
from evaluation_metrics import calculate_inception_score_confidence

path = 'ucf101/mocogan_ode'

if __name__ == '__main__':
    # gen model
    gen = VideoGenerator(3, 50, 0, 16, 16)

    state_dicts = torch.load(f'checkpoints/{path}/state_normal42000.ckpt')

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    gen.cpu()
    results = calculate_inception_score_confidence(gen, test=False, moco=True)
    np.save('results/mocogan ode lin.npy', results)
    print(results)
