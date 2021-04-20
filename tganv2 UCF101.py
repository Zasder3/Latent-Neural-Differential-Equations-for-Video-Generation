import torch
import torch.nn.functional as F
import numpy as np
from skvideo import io
from ucf101.UCF101DatasetTGAN import UCF101
from models.tganv2_gen import Generator_CLSTM
from models.tganv2_dis import DisMultiResNet
from evaluation_metrics import calculate_inception_score
from tqdm import tqdm
from pathlib import Path

epochs = 100000
batch_size = 32
path = 'ucf101/tganv2'
start_epoch = 0
lambda_val = 0.5
conf = ''
dset = ''

Path('video_samples/' + path).mkdir(parents=True, exist_ok=True)
Path('checkpoints/' + path).mkdir(parents=True, exist_ok=True)
Path('epoch_is/').mkdir(parents=True, exist_ok=True)


def genSamples(g, n=8, e=1):
    with torch.no_grad():
        s = g(torch.rand((n**2, 256), device='cuda')*2-1,
              test=True).cpu().detach().numpy()
    out = np.zeros((1, 20, 64*n, 64*n))

    for j in range(n):
        for k in range(n):
            out[:, :, 192*j:192*(j+1), 192*k:192*(k+1)] = s[j*n + k, 0, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (np.concatenate([out, out, out], axis=3)+1) / 2 * 255
    io.vwrite(f'tganv2moving/gensamples_id{e}.gif', out)


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
    test = UCF101(dset, conf, img_size=192)
    loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True,
                                         drop_last=True)

    def dataGen():
        while True:
            for d in loader:
                yield d

    dg = dataGen()
    # gen model
    dis = DisMultiResNet().cuda()
    gen = Generator_CLSTM().cuda()
    disOpt = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0, 0.9))
    genOpt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
    disSched = torch.optim.lr_scheduler.LambdaLR(disOpt, lambda epoch: 1-epoch/epochs)
    genSched = torch.optim.lr_scheduler.LambdaLR(genOpt, lambda epoch: 1-epoch/epochs)

    # train
    # note on loss function: within the current github repo they
    # employ softplus linear loss, if the normal cross entropy
    # is desired one may simply change the comments
    isScores = []
    for epoch in tqdm(range(epochs)):
        # discriminator
        disOpt.zero_grad()
        real = next(dg).cuda()
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
        disSched.step()
        # generator
        genOpt.zero_grad()
        fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
        pf = dis(fake)
        # gen_loss = -torch.mean(torch.log(pf))
        gen_loss = torch.mean(F.softplus(-pf))
        gen_loss.backward()
        genOpt.step()
        genSched.step()
        # log results
        if epoch % 1000 == 0:
            genSamples(gen, e=epoch)
            gen.cpu()
            isScores.append(calculate_inception_score(gen))
            print(isScores[-1])
            np.save('epoch_is/tganv2.npy', isScores)
            gen.cuda()
            torch.save({'epoch': epoch,
                        'model_state_dict': [gen.state_dict(),
                                             dis.state_dict()],
                        'optimizer_state_dict': [genOpt.state_dict(),
                                                 disOpt.state_dict()],
                        'optimizer_schedule_dict': [genSched.state_dict(),
                                                    disSched.state_dict()]},
                       f'checkpoints/{path}/state_normal{epoch}.ckpt')
    gen.cpu()
    isScores.append(calculate_inception_score(gen))
    print(isScores[-1])
    np.save('epoch_is/tganv2.npy', isScores)
    gen.cuda()
    torch.save({'epoch': epoch,
                'model_state_dict': [gen.state_dict(),
                                     dis.state_dict()],
                'optimizer_state_dict': [genOpt.state_dict(),
                                         disOpt.state_dict()],
                'optimizer_schedule_dict': [genSched.state_dict(),
                                            disSched.state_dict()]},
               f'checkpoints/{path}/state_normal{epoch}.ckpt')


if __name__ == '__main__':
    train()