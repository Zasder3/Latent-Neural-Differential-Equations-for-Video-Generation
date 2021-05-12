import torch
import torch.nn as nn
import models.mocogan as mocogan
from torchdiffeq import odeint_adjoint as odeint
from torchsde import sdeint_adjoint as sdeint

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fn = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh()
            )

    def forward(self, t, x):
        return self.fn(x)


class ODEFuncDeep(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()

        self.fn = nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim)
            )

    def forward(self, t, x):
        return self.fn(x)


class SDEFunc(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()

        self.drift_fn = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim)
        )
        self.diffusion_fn = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim)
        )
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, x):
        return self.drift_fn(x)
    
    def g(self, t, x):
        return self.diffusion_fn(x)


class NOODEFunc(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim)
        )

    def forward(self, t, x):
        assert isinstance(x, tuple)
        # highest order derivative
        # z.requires_grad_()
        z = self.fn(x[0])

        return tuple([x[i-1] if i else z for i in range(len(x))])


class VideoGenerator(mocogan.VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=ODEFunc, dim_hidden=None, linear=True):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)
        if dim_hidden:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_hidden)
        else:
            self.ode_fn = ode_fn(dim=dim_z_motion)
        
        if linear:
            self.linear = nn.Sequential(
                    nn.Linear(dim_z_motion, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, dim_z_motion),
                    nn.LeakyReLU(0.2)
                    )
        else:
            self.linear = nn.Identity()

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        x = torch.randn(num_samples, self.dim_z_motion, device='cuda')

        x = self.linear(x)

        z_m_t = odeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='rk4')

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t


class VideoGeneratorNOODE(VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length, dim_hidden=None, order=2):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length, ode_fn=NOODEFunc, dim_hidden=dim_hidden, linear=False)
        self.dsdt_encoders = nn.ModuleList(
            [nn.Linear(dim_z_motion, dim_z_motion) for i in range(order)]
        )

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        x = torch.randn(num_samples, self.dim_z_motion, device='cuda')

        x = tuple([f(x) for f in self.dsdt_encoders])

        z_m_t = odeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='rk4')[-1]

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t


class VideoGeneratorSDE(VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length, dim_hidden=None, linear=True):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length, ode_fn=SDEFunc, dim_hidden=dim_hidden, linear=False)
    
    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        x = torch.randn(num_samples, self.dim_z_motion, device='cuda')

        x = self.linear(x)

        z_m_t = sdeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='euler', adjoint_method='euler', dt=2.5e-2)

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t
