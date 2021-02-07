import torch
import torch.nn as nn
import numpy as np
import math


class CLSTM_cell(nn.Module):
    def __init__(self, n_filters):
        super(CLSTM_cell, self).__init__()
        self.w_x = nn.Conv2d(n_filters, n_filters * 4, kernel_size=3,
                             padding=1)
        self.w_h = nn.Conv2d(n_filters, n_filters * 4, kernel_size=3,
                             padding=1, bias=False)

    def forward(self, x, h=None, c=None):
        xifoc = self.w_x(x)
        xi, xf, xo, xc = xifoc.chunk(4, dim=1)
        if h is not None:
            hi, hf, ho, hc = self.w_h(h).chunk(4, dim=1)
        else:
            hi, hf, ho, hc = torch.zeros_like(xifoc).chunk(4, dim=1)

        if c is None:
            c = torch.zeros_like(x)

        ci = torch.sigmoid(xi + hi)
        cf = torch.sigmoid(xf + hf)
        co = torch.sigmoid(xo + ho)
        cc = cf * c + ci * torch.tanh(xc + hc)
        ch = torch.tanh(cc) * co

        return ch, cc


class CLSTM(nn.Module):
    def __init__(self, n_filters, n_frames):
        super(CLSTM, self).__init__()
        self.cell = CLSTM_cell(n_filters)
        self.n_frames = n_frames

    def forward(self, z):
        # Assume z is in proper convolutional shape
        out = torch.stack([torch.zeros_like(z)]*self.n_frames, dim=1)

        h, c = None, None
        for i in range(self.n_frames):
            h, c = self.cell(z, h, c)
            out[:, i] = h
            z = torch.zeros_like(z)

        return out


class Up(nn.Module):
    def __init__(self, cin, cout):
        super(Up, self).__init__()
        self.relu = nn.ReLU()

        # define main branch
        self.upsample = nn.Upsample(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(cin)
        self.convm1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.convm2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)

        # define skip branch
        self.sconv = nn.Conv2d(cin, cout, kernel_size=1)

        # initialize
        nn.init.xavier_uniform_(self.convm1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.convm2.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.sconv.weight)

    def forward(self, x):
        # compute main
        h = self.bn1(x)
        h = self.relu(h)
        h = self.upsample(h)
        h = self.convm1(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.convm2(h)

        # compute skip
        s = self.upsample(x)
        s = self.sconv(s)

        return h + s


class Render(nn.Module):
    def __init__(self, cin, colors=1):
        super(Render, self).__init__()
        self.bn = nn.BatchNorm2d(cin)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(cin, colors, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = torch.tanh(x)

        return x


class Generator_CLSTM(nn.Module):
    def __init__(self, z_dim=256,
                 tempc=1024,
                 zt_dim=3,
                 upchannels=[512, 256, 128],
                 subchannels=[64, 32, 32],
                 n_frames=16,
                 colors=3,
                 subsample=4):
        super(Generator_CLSTM, self).__init__()
        assert len(subchannels) == 3
        self.tempc = tempc
        self.zt_dim = zt_dim
        self.colors = colors
        self.frames = subsample

        self.fc = nn.Linear(z_dim, zt_dim**2 * tempc)
        self.temp = CLSTM(tempc, n_frames)

        self.build = nn.Sequential()
        for i in range(len(upchannels)):
            if not i:
                self.build.add_module('Up1', Up(tempc, upchannels[0]))
            else:
                self.build.add_module(f'Up{i+1}', Up(upchannels[i-1],
                                      upchannels[i]))

        self.buildr = Render(upchannels[-1], colors=colors)

        self.sup1 = Up(upchannels[-1], subchannels[0])
        self.sup1r = Render(subchannels[0], colors=colors)
        self.sup2 = Up(subchannels[0], subchannels[1])
        self.sup2r = Render(subchannels[1], colors=colors)
        self.sup3 = Up(subchannels[1], subchannels[2])
        self.sup3r = Render(subchannels[2], colors=colors)

        # for models that extend
        self.penalties = None

    def subsample(self, h, N, T):
        # to vid
        _, C, H, W = h.shape
        h = h.view(N, T, C, H, W)
        h = h[:, np.random.randint(min(self.frames, T))::self.frames]
        N, T, C, H, W = h.shape
        # to train
        h = h.contiguous().view(N * T, C, H, W)
        return h, T

    def forward(self, z, test=False):
        h = self.fc(z)
        h = h.view(-1, self.tempc, self.zt_dim, self.zt_dim)
        h = self.temp(h)
        N, T, C, H, W = h.shape
        h = h.view(N*T, C, H, W)
        h = self.build(h)

        outsize = self.zt_dim * 2 ** (len(self.build) + 3)

        if test:
            h = self.sup1(h)
            h = self.sup2(h)
            h = self.sup3(h)
            h = self.sup3r(h).view(N, T, self.colors, outsize,
                                   outsize).transpose(1, 2)

            return h
        else:
            # render 1st
            x1 = self.buildr(h).view(N, T, self.colors, outsize // 8,
                                     outsize // 8)
            h, T = self.subsample(h, N, T)
            h = self.sup1(h)
            # render 2nd
            x2 = self.sup1r(h).view(N, T, self.colors, outsize // 4,
                                    outsize // 4)
            h, T = self.subsample(h, N, T)
            h = self.sup2(h)
            # render 3rd
            x3 = self.sup2r(h).view(N, T, self.colors, outsize // 2,
                                    outsize // 2)
            h, T = self.subsample(h, N, T)
            h = self.sup3(h)
            # render 4th
            x4 = self.sup3r(h).view(N, T, self.colors, outsize, outsize)

        return x1, x2, x3, x4
