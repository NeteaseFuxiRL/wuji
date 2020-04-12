"""
Copyright (C) 2018--2020, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import collections.abc

import torch.nn as nn
import torch.nn.functional as F

from . import Channel, Module
from . import wrap
from .fc import Linear


class Conv2d(nn.Module):
    def __init__(self, inputs, outputs, kernel_size, padding=0, stride=1, bn=False, act='relu'):
        super().__init__()
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(inputs, outputs, kernel_size, stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(outputs, momentum=0.01) if bn else lambda x: x
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


@wrap.group.rsplit(1)
class DNN1(Module):
    def __init__(self, config, inputs, outputs):
        super().__init__()
        self.config = config
        channel = Channel(inputs)
        self.conv = nn.Sequential(
            Conv2d(channel(), channel.next(8), kernel_size=3, padding=True),
            nn.MaxPool2d(kernel_size=2),
            Conv2d(channel(), channel.next(16), kernel_size=3, padding=True),
        )
        self.linear = nn.Sequential(
            Linear(channel(), channel.next(16)),
            Linear(channel(), outputs, act=None),
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.avg_pool2d(x, x.size()[-2:]).view(x.size(0), -1)
        return self.linear(x)


@wrap.group.rsplit(1)
class BlockMaze(Module):
    def __init__(self, config, inputs, outputs, num=400):
        super().__init__()
        self.config = config
        channel = Channel(num)
        # self.linear = nn.Sequential(*[
        #     Conv2d(channel(), channel.next(8), kernel_size=3, stride=2, padding=1),
        #     Conv2d(channel(), channel.next(16), kernel_size=3, stride=2, padding=1),
        # ])
        # channel = Channel(num)

        self.linear = nn.Sequential(*[
            Linear(channel(), channel.next(128)),
            Linear(channel(), channel.next(128)),
            Linear(channel(), channel.next(outputs), act=None),
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
