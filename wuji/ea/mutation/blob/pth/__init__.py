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
import random
import copy
import configparser

import torch

from .. import ENCODING, Mutation as _Mutation


class Gaussian(_Mutation):
    def __init__(self, config):
        self.config = config
        try:
            prob = config.getfloat('mutation_prob', ENCODING)
            self.prob = lambda parent: prob
        except configparser.NoOptionError:
            self.prob = lambda blob: 1 / sum(value.numpy().size for value in blob.values())
        self.stddev = self.config.getfloat('mutation_gaussian', 'stddev')
        self.random = random.Random()
        self.generator = torch.Generator()

    def __call__(self, blob, **kwargs):
        assert isinstance(blob, collections.OrderedDict), type(blob)
        if self.random.random() >= self.prob(blob):
            return copy.deepcopy(blob)
        for key, value in blob.items():
            if value.size():
                blob[key] = value + torch.randn(*value.size(), generator=self.generator) * self.stddev
        return blob
