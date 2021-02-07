import numpy as np
import torch
import mocoganmodels
import torch.nn as nn
from skvideo import io
from torch.optim import Adam
from MovingMNIST import MovingMNIST

g = mocoganmodels.VideoGenerator(1, 256, 0, 256, 20, ngf=64).cuda()

d = mocoganmodels.VideoDiscriminator(1, ndf=64).cuda()

d_i = mocoganmodels.ImageDiscriminator(1, ndf=64).cuda()

test = MovingMNIST('moving/', train=False)
loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True, drop_last=True)

def dataGen():
    while True:
        for d in loader:
            yield d

def genSamples(g, n=8,e=1):
    s = g.sample_videos(n**2)[0].cpu().detach().numpy()
    out = np.zeros((1, 20, 64*n, 64*n))
    
    for j in range(n):
        for k in range(n):
            out[:, :, 64*j:64*(j+1), 64*k:64*(k+1)] = s[j*n + k, : :, :, :]
    
    out = out.transpose((1,2,3,0))
    out = (np.concatenate([out, out, out], axis=3)+1) / 2 * 255
    io.vwrite(f'epochsamples/gensamples_id{e}.gif', out)

g_optim = Adam(g.parameters(), lr=1e-4)
d_optim = Adam(d.parameters(), lr=1e-5)
d_i_optim = Adam(d_i.parameters(), lr=1e-5)
epochs = int(1e5)
gen = dataGen()
loss_fn = nn.BCEWithLogitsLoss()

for e in range(epochs):
    #disc vid
    real = next(gen)
    real = (torch.cat(real, dim=1).unsqueeze(1).type(torch.FloatTensor).cuda()/255-0.5)/0.5
    pr = d(real)[0]
    with torch.no_grad():
        f = g.sample_videos(32)[0]
    pf = d(f)[0]
    ones_v = torch.ones(pr.size(), device='cuda')
    zeros_v = torch.zeros(pr.size(), device='cuda')

    loss_d = loss_fn(pr, ones_v) + loss_fn(pf, zeros_v)
    
    loss_d.backward()
    d_optim.step()
    d_optim.zero_grad()
    d_i_optim.zero_grad()
    g_optim.zero_grad()
    print(e,'DISCV', loss_d.item())
    
    #disc im
    indexes = torch.randint(0, 20, (32, ))
    ims = torch.empty((32, 1, 64, 64), device='cuda')
    
    real = next(gen)
    real = (torch.cat(real, dim=1).unsqueeze(1).type(torch.FloatTensor).cuda()/255-0.5)/0.5
    for i in range(32):
        ims[i] = real[i, :, indexes[i]]
    pr = d_i(ims)[0]
    pf = d_i(g.sample_images(32)[0])[0]
    
    ones_i = torch.ones(pr.size(), device='cuda')
    zeros_i = torch.zeros(pr.size(), device='cuda')
    
    loss_d_i = loss_fn(pr, ones_i) + loss_fn(pf, zeros_i)
    
    loss_d_i.backward()
    d_i_optim.step()
    d_optim.zero_grad()
    d_i_optim.zero_grad()
    g_optim.zero_grad()
    print(e,'DISCI', loss_d_i.item())
    
    #gen
    pf = d(g.sample_videos(32)[0])[0]
    pf_i = d_i(g.sample_images(32)[0])[0]
    loss_v = loss_fn(pf, ones_v)
    loss_i = loss_fn(pf_i, ones_i)
    loss_g = loss_v + loss_i
    loss_g.backward()
    g_optim.step()
    d_optim.zero_grad()
    d_i_optim.zero_grad()
    g_optim.zero_grad()
    print(e, 'GENV', loss_v.item())
    print(e, 'GENI', loss_i.item())
    print(e, loss_g.item())
    
    if e % 50 == 0:
        genSamples(g, e=e)