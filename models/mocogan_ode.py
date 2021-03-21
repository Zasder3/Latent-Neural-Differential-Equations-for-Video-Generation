import torch
import torch.nn as nn
import models.mocogan as mocogan
from torchdiffeq import odeint_adjoint as odeint

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


class VideoGenerator(mocogan.VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=ODEFunc, dim_hidden=None):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)
        if dim_hidden:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_hidden)
        else:
            self.ode_fn = ode_fn(dim=dim_z_motion)
        self.linear = nn.Sequential(
                nn.Linear(dim_z_motion, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, dim_z_motion),
                nn.LeakyReLU(0.2)
                )

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        x = torch.randn(num_samples, self.dim_z_motion, device='cuda')

        x = self.linear(x)

        z_m_t = odeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='rk4').contiguous()

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t
