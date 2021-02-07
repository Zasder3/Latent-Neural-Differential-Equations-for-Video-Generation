import torch
import torch.nn as nn
import math
from torchsde import sdeint_adjoint as sdeint
from models.tganv2_gen import Generator_CLSTM


class SDE(nn.Module):
    def __init__(self, dim):
        """A stochastic differential equation

        Args:
            dim (int): latent dimension
        """
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_drift = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_diffusion = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        nn.init.xavier_uniform_(self.conv_drift.weight, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv_diffusion.weight, math.sqrt(2))

    def f(self, t, x):
        x = self.conv_drift(x)
        x = self.relu(x)

        return x

    def g(self, t, x):
        x = self.conv_diffusion(x)
        x = self.relu(x)

        return x


class SDEGen(nn.Module):
    def __init__(self, num_frames, dim):
        super().__init__()

        self.num_frames = num_frames
        self.dim = dim
        self.func = SDE(dim)

    def forward(self, x):
        x = sdeint(
            self.func,
            x,
            torch.linspace(0, 1, self.num_frames).float(),
            method='euler', adjoint_method='euler', dt=5e-2
        )

        return x.transpose(0, 1).contiguous()


class GeneratorSDE(Generator_CLSTM):
    def __init__(self, z_dim=256,
                 tempc=1024,
                 zt_dim=3,
                 upchannels=[512, 256, 128],
                 subchannels=[64, 32, 32],
                 n_frames=16,
                 colors=3,
                 conv_cat=True):
        super().__init__(z_dim,
                         tempc,
                         zt_dim,
                         upchannels,
                         subchannels,
                         n_frames,
                         colors)
        self.temp = SDEGen(n_frames, tempc)
