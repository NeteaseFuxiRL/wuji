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

import random
import configparser

import numpy as np

from . import ENCODING, Mutation as _Mutation


class Mutation(_Mutation):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        try:
            prob = config.getfloat('mutation_prob', ENCODING)
            self.prob = lambda parent: prob
        except configparser.NoOptionError:
            self.prob = lambda parent: 1 / len(parent)
        boundary = evaluator.context['encoding'][ENCODING]['boundary']
        self.lower, self.upper = boundary.T
        assert np.all(self.lower < self.upper), boundary
        range = self.upper - self.lower + 1
        self.bits = np.ceil(np.log2(range)).astype(range.dtype)
        self.random = random.Random()

    def __call__(self, parent, **kwargs):
        assert len(parent.shape) == 1
        prob = self.prob(parent)
        return np.vectorize(lambda parent, lower, upper, bits: self.mutate(parent, lower, upper, bits, prob))(parent, self.lower, self.upper, self.bits)

    def mutate(self, parent, lower, upper, bits, prob):
        mask = 1
        for i in range(bits):
            if self.random.random() < prob:
                parent ^= mask
            mask <<= 1
        return np.clip(parent, lower, upper)
