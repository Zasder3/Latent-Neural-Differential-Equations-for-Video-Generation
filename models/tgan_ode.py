import torch
import torch.nn as nn
import models.tgan as tgan
from torchdiffeq import odeint_adjoint as odeint
from torchsde import sdeint_adjoint as sdeint


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
        nn.init.xavier_uniform_(self.fn[2].weight, gain=2**0.5)

    def forward(self, t, x):
        return self.fn(x)


class SDEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.drift_fn = nn.Sequential(
            nn.Linear(100, dim),
            nn.Tanh()
        )
        self.diffusion_fn = nn.Sequential(
            nn.Linear(100, dim),
            nn.Tanh()
        )
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        nn.init.xavier_uniform_(self.drift_fn[0].weight, gain=2**0.5)
        nn.init.xavier_uniform_(self.diffusion_fn[0].weight, gain=2**0.5)
    
    def f(self, t, x):
        return self.drift_fn(x)
    
    def g(self, t, x):
        return self.diffusion_fn(x)


class NOODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fn = nn.Sequential(
                nn.Linear(100, dim),
                nn.Tanh()
            )
        nn.init.xavier_uniform_(self.fn[0].weight, gain=2**0.5)

    def forward(self, t, x):
        assert isinstance(x, tuple)
        # highest order derivative
        # z.requires_grad_()
        z = self.fn(x[0])

        return tuple([x[i-1] if i else z for i in range(len(x))])


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


class TemporalGeneratorSDE(TemporalGeneratorODE):
    def __init__(self, num_frames=16, dim=100, linear=False, ode_fn=SDEFunc):
        super().__init__(num_frames=num_frames, dim=dim, linear=linear, ode_fn=SDEFunc)
        self.ts = torch.linspace(0, 1, self.num_frames).float().cuda()

    def forward(self, x):
        if self.do_linear:
            x = self.linear(x)
        x = sdeint(self.ode_fn, x, self.ts,
                   method='euler', adjoint_method='euler', dt=2.5e-2)

        return x.transpose(0, 1)


class TemporalGeneratorNOODE(TemporalGeneratorODE):
    def __init__(self, num_frames=16, dim=100, linear=False, ode_fn=NOODEFunc, order=2):
        super().__init__(num_frames=num_frames, dim=dim, linear=linear, ode_fn=ode_fn)
        self.dsdt_encoders = nn.ModuleList(
            [nn.Linear(dim, dim) for i in range(order)]
        )
        self.dsdt_encoders.apply(self.init_linear_weights)

    def forward(self, x):
        if self.do_linear:
            x = self.linear(x)
        x = tuple([f(x) for f in self.dsdt_encoders])
        x = odeint(self.ode_fn, x,
                   torch.linspace(0, 1, self.num_frames).float(),
                   method='rk4')

        return x[-1].transpose(0, 1)


class VideoGenerator(tgan.VideoGenerator):
    def __init__(self, dim=100, linear=False, ode_fn=ODEFunc):
        super().__init__()
        self.temp = TemporalGeneratorODE(dim=dim, linear=linear, ode_fn=ode_fn)


class VideoGeneratorSDE(tgan.VideoGenerator):
    def __init__(self, dim=100, linear=False):
        super().__init__()
        self.temp = TemporalGeneratorSDE(dim=dim, linear=linear)


class VideoGeneratorNOODE(tgan.VideoGenerator):
    def __init__(self, dim=100, linear=False, order=2):
        super().__init__()
        self.temp = TemporalGeneratorNOODE(dim=dim, linear=linear, order=order)