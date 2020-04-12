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

import collections

import torch.nn as nn

from . import wrap


class Channel(object):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self):
        return self.channels

    def set(self, channels):
        self.channels = channels

    def next(self, channels):
        self.channels = channels
        return self.channels


class Module(nn.Module):
    def set_blob(self, state_dict):
        return self.load_state_dict(state_dict)

    def get_blob(self):
        return collections.OrderedDict([(key, var.cpu()) for key, var in super().state_dict().items()])
