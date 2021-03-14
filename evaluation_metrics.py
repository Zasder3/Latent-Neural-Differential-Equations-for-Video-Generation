import torch
import cupy as cp
import numpy as np
import chainer
import cv2
from evaluation.c3d_ft import C3DVersion1
from tqdm import tqdm


def calculate_inception_score(gen, n_samples=2048, batch_size=32, test=True, zdim=256, moco=False, reuse=False):
    # generate samples
    batches = n_samples // batch_size
    if not reuse:
        gen = gen.cuda()
        gen.eval()
        for i in range(batches):
            with torch.no_grad():
                z = torch.rand((batch_size, zdim), device='cuda')*2-1
                if test:
                    s = gen(z, test=True).cpu().detach().numpy()
                elif moco:
                    s = gen.sample_videos(batch_size)[0].cpu().detach().numpy()
                else:
                    s = gen(z).cpu().detach().numpy()
            np.save(f'temp/vid{i}.npy', s)
        gen.train()
    del gen
    torch.cuda.empty_cache()

    # load calc model
    dir = 'evaluation/pfnet/chainer/models/conv3d_deepnetA_ucf.npz'
    model = C3DVersion1(pretrained_model=dir).to_gpu()
    mean = np.load('evaluation/mean2.npz')['mean'].astype('f')
    mean = mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]

    # calc probabilities
    out = []
    for i in range(batches):
        x = np.load(f'temp/vid{i}.npy')
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
            out.append(model(x)['prob'].data.get())

    del model
    del x

    out = np.concatenate(out, axis=0)
    np.save('temp/out.npy', out)
    assert out.shape[0] == n_samples
    print(out.shape)

    # find score
    eps = 1e-7
    p_marginal = np.mean(out, axis=0)

    kl = out * (np.log(out + eps) - np.log(p_marginal + eps))
    kl = np.mean(np.sum(kl, axis=1))

    kl = np.exp(kl)

    return kl


def calculate_inception_score_confidence(gen, n_samples=2048, batch_size=32,
                                         iterations=10, test=True, zdim=256,
                                         reuse=False):
    scores = np.empty([iterations])
    for i in tqdm(range(iterations)):
        scores[i] = calculate_inception_score(gen, n_samples, batch_size, test, zdim, reuse)
        print(scores[i])

    return scores.mean(), scores.std()
