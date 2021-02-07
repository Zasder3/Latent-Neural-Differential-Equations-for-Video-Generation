import torch
import torch.nn as nn
import models.tgan as tgan
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, dim=100):
        super().__init__()

        self.fn = nn.Sequential(
                nn.Linear(100, 100),
                nn.Tanh()
            )
        nn.init.xavier_uniform_(self.fn[0].weight, gain=2**0.5)

    def forward(self, t, x):
        return self.fn(x)


class ODEFuncDeep(nn.Module):
    def __init__(self, dim=100):
        super().__init__()

        self.fn = nn.Sequential(
                nn.Linear(100, dim),
                nn.Tanh(),
                nn.Linear(dim, 100)
            )
        nn.init.xavier_uniform_(self.fn[0].weight, gain=2**0.5)

    def forward(self, t, x):
        return self.fn(x)


class TemporalGeneratorODE(nn.Module):
    def __init__(self, num_frames=16, dim=100, linear=False, ode_fn=ODEFunc):
        super().__init__()
        self.ode_fn = ode_fn(dim=dim)
        self.num_frames = num_frames
        self.do_linear = linear
        if linear:
            self.linear = nn.Sequential(
                    nn.Linear(100, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 100),
                    nn.LeakyReLU(0.2)
                    )

            self.linear.apply(self.init_linear_weights)

    def init_linear_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)
    def forward(self, x):
        if self.do_linear:
            x = self.linear(x)
        x = odeint(self.ode_fn, x,
                   torch.linspace(0, 1, self.num_frames).float(),
                   method='rk4')

        return x.transpose(0, 1)


class VideoGenerator(tgan.VideoGenerator):
    def __init__(self, dim=100, linear=False, ode_fn=ODEFunc):
        super().__init__()
        self.temp = TemporalGeneratorODE(dim=dim, linear=linear, ode_fn=ode_fn)
