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

import os
import random
import types
import configparser
import unittest

import numpy as np

import wuji
from . import ENCODING, Crossover as _Crossover


class Crossover(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        boundary = evaluator.context['encoding'][ENCODING]['boundary']
        self.lower, self.upper = boundary.T
        assert np.all(self.lower < self.upper), boundary
        range = self.upper - self.lower + 1
        self.bits = np.ceil(np.log2(range)).astype(range.dtype)
        self.mask = np.power(2, self.bits) - 1
        assert self.mask.dtype == range.dtype
        assert np.all(np.greater_equal(self.mask, self.upper - self.lower))
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        assert len(parent1.shape) == 1
        assert parent1.shape == parent2.shape
        assert parent1.dtype == self.mask.dtype, (parent1.dtype, self.mask.dtype)
        assert parent2.dtype == self.mask.dtype, (parent2.dtype, self.mask.dtype)
        return np.vectorize(lambda parent1, parent2, lower, upper, bits, mask: self.crossover(parent1, parent2, lower, upper, bits, mask))(parent1, parent2, self.lower, self.upper, self.bits, self.mask)

    def crossover(self, parent1, parent2, lower, upper, bits, mask):
        position = self.random.randint(0, bits - 1)
        mask_upper = mask << position
        mask_lower = ~mask_upper
        assert lower <= parent1 <= upper, (lower, parent1, upper)
        assert lower <= parent2 <= upper, (lower, parent2, upper)
        _parent1 = parent1 - lower
        _parent2 = parent2 - lower
        _child1 = (_parent1 & mask_upper) | (_parent2 & mask_lower)
        _child2 = (_parent1 & mask_lower) | (_parent2 & mask_upper)
        child1 = np.clip(lower + _child1, lower, upper)
        child2 = np.clip(lower + _child2, lower, upper)
        if self.random.random() < 0.5:
            return child1, child2
        else:
            return child2, child1


class Test(unittest.TestCase):
    def setUp(self):
        self.config = configparser.ConfigParser()
        wuji.config.load(self.config, os.sep.join(__file__.split(os.sep)[:-5] + ['config.ini']))
        evaluator = types.SimpleNamespace()
        evaluator.context = dict(encoding=dict(integer=dict(boundary=np.array([(-3, 5), (-1, 4)], np.int))))
        self.crossover = Crossover(self.config, evaluator)
        self.parent1 = np.array([-2, 3], np.int)
        self.parent2 = np.array([3, 0], np.int)

    def test_same(self):
        child1, child2 = self.crossover(self.parent1, self.parent1)
        np.testing.assert_array_almost_equal(self.parent1, child1)
        np.testing.assert_array_almost_equal(self.parent1, child2)

    def test_different(self):
        self.crossover(self.parent1, self.parent2)


if __name__ == '__main__':
    unittest.main()
