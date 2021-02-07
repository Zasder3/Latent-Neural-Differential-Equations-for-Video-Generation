import torch
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

    state_dicts = torch.load(f'D:/model_code/checkpoints/{path}/state_normal87000.ckpt')

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    gen.cpu()
    perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    for i, p in enumerate(tqdm(perms)):
        results = calculate_inception_score(gen, perm=p, reuse=i!=0)
        print(results, p)
