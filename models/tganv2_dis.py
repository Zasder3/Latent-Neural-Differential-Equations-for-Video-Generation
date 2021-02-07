import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OptimizedDiscBlock(nn.Module):
    def __init__(self, cin, cout):
        super(OptimizedDiscBlock, self).__init__()
        self.c1 = nn.Conv3d(cin, cout, kernel_size=3, padding=1)
        self.c2 = nn.Conv3d(cout, cout, kernel_size=3, padding=1)
        self.c_sc = nn.Conv3d(cin, cout, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.avgp2d = nn.AvgPool3d(kernel_size=(1, 2, 2))

        # init
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c_sc.weight)

    def forward(self, x):
        h = self.c1(x)
        h = self.relu(h)
        h = self.c2(h)
        h = self.avgp2d(h)

        s = self.avgp2d(x)
        s = self.c_sc(s)

        return h + s


class DisBlock(nn.Module):
    def __init__(self, cin, cout):
        super(DisBlock, self).__init__()
        self.c1 = nn.Conv3d(cin, cin, kernel_size=3, padding=1)
        self.c2 = nn.Conv3d(cin, cout, kernel_size=3, padding=1)
        self.s_sc = nn.Conv3d(cin, cout, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

        # init
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.s_sc.weight)

    def downsample(self, x):
        ksize = [(2 if 1 < k else 1) for k in x.shape[2:]]
        pad = [(0 if k % 2 == 0 else 1) for k in x.shape[2:]][::-1]
        padf = []
        for p in pad:
            padf.append(p)
            padf.append(p)
        x = F.pad(x, padf)
        return F.avg_pool3d(x, kernel_size=ksize, padding=0)

    def forward(self, x):
        h = self.relu(x)
        h = self.c1(h)
        h = self.relu(h)
        h = self.c2(h)
        h = self.downsample(h)

        s = self.s_sc(x)
        s = self.downsample(s)
        return h + s


class DisResNet(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512, 1024], colors=3):
        super(DisResNet, self).__init__()
        self.convs = nn.Sequential()
        self.colors = colors

        for i in range(len(channels)):
            if not i:
                self.convs.add_module(
                    'OptDisc',
                    OptimizedDiscBlock(colors, channels[0])
                )
            else:
                self.convs.add_module(
                    f'Down{i}',
                    DisBlock(channels[i-1], channels[i])
                )

        self.fc = nn.Linear(channels[-1], 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        if x.shape[2] == self.colors:
            x = x.transpose(1, 2)
        h = self.convs(x)
        h = torch.sum(h, dim=(2, 3, 4))
        h = self.fc(h)

        return h


class DisMultiResNet(nn.Module):
    def __init__(self, layers=4, channels=[64, 128, 256, 512, 1024], colors=3):
        super(DisMultiResNet, self).__init__()
        self.layers = layers
        self.res = nn.ModuleList(
            [DisResNet(channels, colors) for _ in range(layers)]
        )

    def forward(self, x):
        assert self.layers == len(x)
        out = [self.res[i](x[i]) for i in range(self.layers)]
        out = torch.cat(out, dim=0)
        # out = sum(out)
        # out = torch.sigmoid(out)

        return out
