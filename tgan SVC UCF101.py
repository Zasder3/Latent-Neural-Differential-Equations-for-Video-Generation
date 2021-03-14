import torch
import torch.nn as nn
import numpy as np
from skvideo import io
from ucf101.UCF101DatasetTGAN import UCF101
from models.tgan import VideoGenerator, VideoDiscriminator
from evaluation_metrics import calculate_inception_score
from tqdm.gui import tqdm

epochs = 100000
batch_size = 32
path = 'ucf101/tgan_svc'
start_epoch = 0
conf = "C:/Video Datasets/UCF101_tgan/ucf101_train_pd.pkl"
dset = "C:/Video Datasets/ucf101_64px_tgan/train.h5"


def singular_value_clip(w):
    dim = w.shape
    if len(dim) > 2:
        w = w.reshape(dim[0], -1)
    u, s, v = torch.svd(w, some=True)
    s[s > 1] = 1
    return (u @ torch.diag(s) @ v.t()).view(dim)


def genSamples(g, n=8, e=1):
    g.eval()
    with torch.no_grad():
        s = g(torch.rand((n**2, 100), device='cuda')*2-1).cpu().detach().numpy()
    g.train()

    out = np.zeros((3, 16, 64*n, 64*n))

    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n + k, :, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (out + 1) / 2 * 255
    io.vwrite(
        f'video_samples/{path}/gensamples_id{e}.gif',
        out
    )


def train():
    # data
    test = UCF101(dset, conf)
    loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)

    def dataGen():
        while True:
            for d in loader:
                yield d

    dg = dataGen()
    # gen model
    dis = VideoDiscriminator().cuda()
    gen = VideoGenerator().cuda()
    disOpt = torch.optim.RMSprop(dis.parameters(), lr=5e-5)
    genOpt = torch.optim.RMSprop(gen.parameters(), lr=5e-5)

    # resume training
    state_dicts = torch.load(f'checkpoints/{path}/state_normal99999.ckpt')
    start_epoch = state_dicts['epoch'] + 1

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    dis.load_state_dict(state_dicts['model_state_dict'][1])
    genOpt.load_state_dict(state_dicts['optimizer_state_dict'][0])
    disOpt.load_state_dict(state_dicts['optimizer_state_dict'][1])
    # train
    # isScores = []
    isScores = list(np.load('tgan_svc_inception.npy'))
    for epoch in tqdm(range(start_epoch, epochs)):
        assert gen.training
        # discriminator
        disOpt.zero_grad()
        real = next(dg).cuda()
        real = real.to(dtype=torch.float32).transpose(1, 2)

        pr = dis(real)
        with torch.no_grad():
            fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
        pf = dis(fake)
        # dis_loss = -torch.mean(torch.log(pr) + torch.log(1-pf))
        dis_loss = torch.mean(-pr) + torch.mean(pf)
        dis_loss.backward()
        disOpt.step()
        # generator
        genOpt.zero_grad()
        fake = gen(torch.rand((batch_size, 100), device='cuda')*2-1)
        pf = dis(fake)
        # gen_loss = -torch.mean(torch.log(pf))
        gen_loss = torch.mean(-pf)
        gen_loss.backward()
        genOpt.step()
        # log results and clip svds
        print('Epoch', epoch, 'Dis', dis_loss.item(), 'Gen', gen_loss.item())
        if epoch % 5 == 0:
            for module in list(dis.model3d.children()) + [dis.conv2d]:
                # discriminator only contains Conv3d, BatchNorm3d, and ReLU
                if type(module) == nn.Conv3d or type(module) == nn.Conv2d:
                    module.weight.data = singular_value_clip(module.weight)
                elif type(module) == nn.BatchNorm3d:
                    gamma = module.weight.data
                    std = torch.sqrt(module.running_var)
                    gamma[gamma > std] = std[gamma > std]
                    gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                    module.weight.data = gamma
            
            if epoch % 100 == 0:
                genSamples(gen, e=epoch)
                if epoch % 1000 == 0:
                    gen.cpu()
                    isScores.append(calculate_inception_score(gen, zdim=100,
                                                              test=False))
                    print(isScores[-1])
                    np.save('tgan_svc_inception.npy', isScores)
                    gen.cuda()
                    torch.save({'epoch': epoch,
                                'model_state_dict': [gen.state_dict(),
                                                    dis.state_dict()],
                                'optimizer_state_dict': [genOpt.state_dict(),
                                                        disOpt.state_dict()]},
                            f'checkpoints/{path}/state_normal{epoch}.ckpt')
    gen.cpu()
    isScores.append(calculate_inception_score(gen,zdim=100, test=False))
    print(isScores[-1])
    np.save('tgan_svc_inception.npy', isScores)
    gen.cuda()
    # torch.save({'epoch': epoch,
    #             'model_state_dict': [gen.state_dict(),
    #                                  dis.state_dict()],
    #             'optimizer_state_dict': [genOpt.state_dict(),
    #                                      disOpt.state_dict()]},
    #            f'checkpoints/{path}/state_normal{epoch}.ckpt')


if __name__ == '__main__':
    train()
