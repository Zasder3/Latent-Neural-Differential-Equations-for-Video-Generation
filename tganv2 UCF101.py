import torch
import torch.nn.functional as F
import numpy as np
from skvideo import io
from ucf101.UCF101Dataset import UCF101
from models.tganv2_gen import Generator_CLSTM
from models.tganv2_dis import DisMultiResNet
from tqdm.gui import tqdm

epochs = 100000
batch_size = 32
lambda_val = 0.5
path = 'ucf101/tganv2'
start_epoch = 0


def genSamples(g, n=8, e=1):
    with torch.no_grad():
        s = g(torch.rand((n**2, 256), device='cuda')*2-1,
              test=True).cpu().detach().numpy()

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


def subsample_real(h, frames=4):
    h = h[:, np.random.randint(min(frames, h.shape[1]))::frames]
    return h


def full_subsample_real(h, frames=4):
    out = []
    for i in range(4):
        if i:
            out.append(subsample_real(out[i-1], frames=frames))
        else:
            out.append(h)

    for i in range(4):
        for j in range(3-i):
            out[i] = F.avg_pool3d(out[i], kernel_size=(1, 2, 2))
    return out


def zero_centered_gp(real_data, pr):
    gradients = torch.autograd.grad(outputs=pr, inputs=real_data,
                                    grad_outputs=torch.ones_like(pr),
                                    create_graph=True, retain_graph=True)

    return sum([torch.sum(torch.square(g)) for g in gradients])


def train():
    # data
    test = UCF101()
    loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)

    def dataGen():
        while True:
            for d in loader:
                yield d

    dg = dataGen()
    # gen model
    dis = DisMultiResNet(channels=[32, 64, 128, 256, 512], colors=3).cuda()
    gen = Generator_CLSTM(
        tempc=512,
        zt_dim=4,
        upchannels=[256],
        subchannels=[128, 64, 32],
        n_frames=16,
        colors=3
    ).cuda()
    disOpt = torch.optim.Adam(dis.parameters(), lr=5e-5, betas=(0, 0.9))
    genOpt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
    # resume training
    state_dicts = torch.load(f'checkpoints/{path}/state_normal97000.ckpt')
    start_epoch = state_dicts['epoch'] + 1

    gen.load_state_dict(state_dicts['model_state_dict'][0])
    dis.load_state_dict(state_dicts['model_state_dict'][1])
    genOpt.load_state_dict(state_dicts['optimizer_state_dict'][0])
    disOpt.load_state_dict(state_dicts['optimizer_state_dict'][1])
    # train
    for epoch in tqdm(range(start_epoch, epochs)):
        # discriminator
        disOpt.zero_grad()
        real, _ = next(dg)
        real = real.cuda()
        real = real.to(dtype=torch.float32)
        real = full_subsample_real(real)
        for i in real:
            i.requires_grad = True
        pr = dis(real)
        dis_loss = zero_centered_gp(real, pr) * lambda_val
        with torch.no_grad():
            fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
        pf = dis(fake)
        # dis_loss = -torch.mean(torch.log(pr) + torch.log(1-pf))
        dis_loss += torch.mean(F.softplus(-pr)) + torch.mean(F.softplus(pf))
        dis_loss.backward()
        disOpt.step()
        # generator
        genOpt.zero_grad()
        fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
        pf = dis(fake)
        # gen_loss = -torch.mean(torch.log(pf))
        gen_loss = torch.mean(F.softplus(-pf))
        gen_loss.backward()
        genOpt.step()
        # log results
        print('Epoch', epoch, 'Dis', dis_loss.item(), 'Gen', gen_loss.item())
        if epoch % 100 == 0:
            genSamples(gen, e=epoch)
            if epoch % 1000 == 0 or epoch == epochs-1:
                torch.save({'epoch': epoch,
                            'model_state_dict': [gen.state_dict(),
                                                 dis.state_dict()],
                            'optimizer_state_dict': [genOpt.state_dict(),
                                                     disOpt.state_dict()]},
                           f'checkpoints/{path}/state_normal{epoch}.ckpt')
    torch.save({'epoch': epoch,
                'model_state_dict': [gen.state_dict(),
                                     dis.state_dict()],
                'optimizer_state_dict': [genOpt.state_dict(),
                                         disOpt.state_dict()]},
               f'checkpoints/{path}/state_normal{epoch}.ckpt')


if __name__ == '__main__':
    train()
