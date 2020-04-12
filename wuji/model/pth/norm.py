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

import numpy as np
import torch
import torch.nn as nn


def _l2(weight, std=1):
    shape = weight.shape
    weight = np.reshape(weight, [shape[0], -1])
    weight *= std / np.sqrt(np.square(weight).sum(-1, keepdims=True))
    weight = np.reshape(weight, shape)
    return weight


def l2(weight, std=1):
    t = torch.from_numpy(_l2(np.random.randn(*weight.shape).astype(np.float32), std))
    return nn.Parameter(t).type_as(weight)


def uniform(weight):
    r = 1 / np.sqrt(np.multiply.reduce(weight.shape[1:]))
    return nn.init.uniform_(weight, -r, r)
