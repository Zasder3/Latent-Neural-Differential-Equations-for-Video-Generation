import torch
import torch.nn as nn
import math
from torchdiffeq import odeint_adjoint as odeint
from models.tganv2_gen import Generator_CLSTM


class ODEFunc(nn.Module):
    def __init__(self, dim, dim_hidden=None):
        """Acts as the equivalent to an RNN cell but for an ODE

        Args:
            dim (int): Latent Channels for ODE
            conv_cat (bool, optional): Concatenate time. Defaults to True.
        """
        super().__init__()
        dim_hidden = dim_hidden if dim_hidden else dim
        self.relu = nn.ReLU() # We need a continuously diff func
        self.fn = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
            )
        nn.init.xavier_uniform_(self.fn[0].weight, math.sqrt(2))
        if dim_hidden != dim:
            self.fn.add_module('conv2', nn.Conv2d(dim_hidden, dim, kernel_size=3, padding=1))
            self.fn.add_module('leakyRelu2', nn.LeakyReLU(0.2))
            nn.init.xavier_uniform_(self.fn[2].weight, math.sqrt(2))

    def forward(self, t, x):
        x.requires_grad_()
        x = self.fn(x)

        return x


class NOODEFunc(nn.Module):
    def __init__(self, dim, conv_cat=True):
        """Acts as the equivalent to an RNN cell but for an
        Nth order ODE

        Args:
            dim (int): Latent Channels for ODE
            conv_cat (bool, optional): Concatenate time. Defaults to True.
        """
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_cat = conv_cat
        add = 1 if conv_cat else 0
        self.conv = nn.Conv2d(dim + add, dim, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv.weight, math.sqrt(2))

    def forward(self, t, x):
        assert isinstance(x, tuple)
        # highest order derivative
        N, C, H, W = x[0].shape
        if self.conv_cat:
            conc = torch.ones((N, 1, H, W), device=x[0].device,
                              requires_grad=True)
            z = torch.cat([conc, x[0]], dim=1)
        z.requires_grad_()
        z = self.conv(z)
        z = self.relu(z)

        return tuple([x[i-1] if i else z for i in range(len(x))])


class ODEGen(nn.Module):
    def __init__(self, num_frames, dim, dim_hidden=None, int_t=1):
        """Acts as the full ODE solver for the temporal latent space

        Args:
            num_frames (int): Number of frames
            dim (int): Latent channels
            conv_cat (bool, optional): Concatenate time. Defaults to True.
        """
        super().__init__()
        self.num_frames = num_frames
        self.dim = dim
        self.ode_fn = ODEFunc(dim, dim_hidden)
        self.int_t = int_t

    def forward(self, x):
        x = odeint(self.ode_fn, x,
                   torch.linspace(0, self.int_t, self.num_frames).float(),
                   method='rk4')

        x = x[-1] if isinstance(x, tuple) else x
        return x.transpose(0, 1).contiguous()


class NOODEGen(ODEGen):
    def __init__(self, num_frames, dim, conv_cat=True, int_t=1):
        """Acts as the full ODE solver for the Nth order temporal
        latent space

        Args:
            num_frames (int): Number of frames
            dim (int): Latent channels
            conv_cat (bool, optional): Concatenate time. Defaults to True.
        """
        super().__init__(num_frames, dim, conv_cat, int_t)
        self.ode_fn = NOODEFunc(dim, conv_cat)


class GeneratorODE(Generator_CLSTM):
    def __init__(self, z_dim=256,
                 tempc=1024,
                 zt_dim=3,
                 upchannels=[512, 256, 128],
                 subchannels=[64, 32, 32],
                 n_frames=16,
                 colors=3,
                 subsample=4,
                 dim_hidden=None,
                 int_t=1):
        super().__init__(z_dim,
                         tempc,
                         zt_dim,
                         upchannels,
                         subchannels,
                         n_frames,
                         colors,
                         subsample)
        self.temp = ODEGen(n_frames, tempc, dim_hidden, int_t)


class GeneratorNOODE(Generator_CLSTM):
    def __init__(self, z_dim=256,
                 tempc=1024,
                 zt_dim=3,
                 upchannels=[512, 256, 128],
                 subchannels=[64, 32, 32],
                 n_frames=16,
                 colors=3,
                 subsample=4,
                 dim_hidden=None,
                 int_t=1,
                 order=2):
        super().__init__(z_dim,
                         tempc,
                         zt_dim,
                         upchannels,
                         subchannels,
                         n_frames,
                         colors,
                         subsample=4)
        self.temp = NOODEGen(n_frames, tempc, dim_hidden, int_t)
        del self.fc
        self.dsdt_encoders = nn.ModuleList(
            [nn.Linear(z_dim, zt_dim**2*tempc) for i in range(order)]
        )

    def forward(self, z, test=False):
        h = [f(z) for f in self.dsdt_encoders]
        h = [i.view(-1, self.tempc, self.zt_dim, self.zt_dim) for i in h]
        h = self.temp(tuple(h))
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
