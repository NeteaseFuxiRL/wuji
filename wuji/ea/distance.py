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

import numpy as np
import torch


def euclidean(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def layerwise_mean(state_dict1, state_dict2):
    assert isinstance(state_dict1, collections.OrderedDict), type(state_dict1)
    assert isinstance(state_dict2, collections.OrderedDict), type(state_dict2)
    return np.mean([torch.sqrt(torch.sum((layer1 - layer2) ** 2)) for layer1, layer2 in zip(state_dict1.values(), state_dict2.values())])
