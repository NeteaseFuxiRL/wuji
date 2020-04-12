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

from wuji.model.pth import Channel
from wuji.model.pth.fc import Linear


def hidden0(module):
    class Module(module):
        def __init__(self, config, inputs, outputs, index=-1):
            super().__init__(config, inputs, outputs)
            self.index = index
            channel = Channel(self.linear[self.index].linear.weight.size(-1))
            self.critic = nn.Sequential(
                Linear(channel(), 1, act=None),
            )
            for module in self.critic.modules():
                self.init(module)

        def forward(self, x):
            share = self.linear[:self.index](x)
            return self.linear[self.index:](share), self.critic(share)
    return Module


def hidden1(module):
    class Module(module):
        def __init__(self, config, inputs, outputs, index=-1):
            super().__init__(config, inputs, outputs)
            self.index = index
            channel = Channel(self.linear[self.index].linear.weight.size(-1))
            self.critic = nn.Sequential(
                Linear(channel(), channel()),
                Linear(channel(), 1, act=None),
            )
            for module in self.critic.modules():
                self.init(module)

        def forward(self, x):
            share = self.linear[:self.index](x)
            return self.linear[self.index:](share), self.critic(share)
    return Module


def hidden2(module):
    class Module(module):
        def __init__(self, config, inputs, outputs, index=-1):
            super().__init__(config, inputs, outputs)
            self.index = index
            channel = Channel(self.linear[self.index].linear.weight.size(-1))
            self.critic = nn.Sequential(
                Linear(channel(), channel()),
                Linear(channel(), channel.next(64)),
                Linear(channel(), channel.next(32)),
                Linear(channel(), 1, act=None),
            )
            for module in self.critic.modules():
                self.init(module)

        def forward(self, x):
            share = self.linear[:self.index](x)
            return self.linear[self.index:](share), self.critic(share)
    return Module
