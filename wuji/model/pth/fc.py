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

import torch.nn as nn

from . import Channel, Module
from . import wrap


def identity(x):
    return x


class Linear(nn.Module):
    def __init__(self, inputs, outputs, norm=None, act='relu', **kwargs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, bias=norm is None or kwargs.get('bias', False))
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(outputs)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(outputs)
        else:
            self.norm = identity
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(kwargs['slope'] if 'slope' in kwargs else 0.1, inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = identity

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x


@wrap.group.rsplit(1)
class DNN0(Module):
    def __init__(self, config, inputs, outputs):
        super().__init__()
        self.config = config
        channel = Channel(inputs)
        self.linear = nn.Sequential(
            Linear(channel(), outputs, act=None),
        )

    def forward(self, x):
        return self.linear(x)


@wrap.group.rsplit(1)
class DNN1(Module):
    def __init__(self, config, inputs, outputs, norm=None):
        super().__init__()
        self.config = config
        channel = Channel(inputs)
        self.linear = nn.Sequential(
            Linear(channel(), channel.next(64), norm=norm),
            Linear(channel(), outputs, act=None),
        )

    def forward(self, x):
        return self.linear(x)


@wrap.group.rsplit(1)
class DNN2(Module):
    def __init__(self, config, inputs, outputs, norm=None):
        super().__init__()
        self.config = config
        channel = Channel(inputs)
        self.linear = nn.Sequential(
            Linear(channel(), channel.next(128), norm=norm),
            Linear(channel(), channel.next(64), norm=norm),
            Linear(channel(), outputs, act=None),
        )

    def forward(self, x):
        return self.linear(x)


@wrap.group.rsplit(1)
class DNN3(Module):
    def __init__(self, config, inputs, outputs, norm=None):
        super().__init__()
        self.config = config
        channel = Channel(inputs)
        self.linear = nn.Sequential(
            Linear(channel(), channel.next(256), norm=norm),
            Linear(channel(), channel.next(128), norm=norm),
            Linear(channel(), channel.next(128), norm=norm),
            Linear(channel(), outputs, act=None),
        )

    def forward(self, x):
        return self.linear(x)
