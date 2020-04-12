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
import itertools

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


class Agent(object):
    @staticmethod
    def numpy(t):
        return t.detach().cpu().numpy()

    @staticmethod
    def serialize(state_dict):
        return bytes(itertools.chain(*[value.cpu().numpy().tostring() for value in state_dict.values()]))

    @staticmethod
    def nbytes(state_dict):
        return sum(value.cpu().numpy().nbytes for value in state_dict.values())

    @staticmethod
    def flat(state_dict):
        return torch.cat([layer.view(-1) for layer in state_dict.values()]).numpy()

    @staticmethod
    def unflat(flat, state_dict):
        layers = []
        begin = 0
        for key, var in state_dict.items():
            end = begin + np.multiply.reduce(var.size())
            layers.append((key, torch.from_numpy(flat[begin:end].reshape(var.size())).to(var.device)))
            begin = end
        return collections.OrderedDict(layers)

    @staticmethod
    def keys(state_dict):
        return list(state_dict.keys())

    @staticmethod
    def values(state_dict):
        return [t.numpy() for t in state_dict.values()]


class Model(Agent):
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def close(self):
        delattr(self, 'model')
        delattr(self, 'device')

    def tensor(self, a, expand=None):
        t = to_tensor(a) if len(a.shape) > 2 else torch.from_numpy(a)
        t = t.to(self.device)
        if expand is not None:
            t = t.unsqueeze(expand)
        return t

    def eval(self):
        self.model.eval()
