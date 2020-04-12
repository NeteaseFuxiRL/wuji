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

"""
@ARTICLE{
  author = {Kalyanmoy Deb and Ram Bhusan Agrawal},
  title = {Simulated Binary Crossover for Continuous Search Space},
  journal = {Complex Systems},
  year = {1994},
  volume = {9},
  pages = {115-148},
  number = {2}
}
"""

import os
import random
import types
import copy
import configparser
import unittest

import numpy as np

import wuji
from . import ENCODING, Crossover as _Crossover


def spread_factor_probability(spread_factor, distribution_index):
    assert spread_factor >= 0, spread_factor
    if spread_factor < 1:
        return (distribution_index + 1) * np.power(spread_factor, distribution_index) / 2
    else:
        return (distribution_index + 1) / np.power(spread_factor, distribution_index + 2) / 2


def amplification_factor(spread_factor, distribution_index):
    assert np.all(spread_factor >= 0), spread_factor
    assert distribution_index >= 0
    return 2 / (2 - np.power(spread_factor, -(distribution_index + 1)))


def spread_factor_instance(amplification_factor, distribution_index, u):
    assert amplification_factor >= 1, amplification_factor
    assert distribution_index >= 0, distribution_index
    assert 0 <= u < 1, u
    if u < amplification_factor / 2:
        return np.power(2 * u / amplification_factor, 1. / (distribution_index + 1))
    else:
        return np.power(1 / (2 - 2 * u / amplification_factor), 1. / (distribution_index + 1))


class Crossover(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        boundary = evaluator.context['encoding'][ENCODING]['boundary']
        self.lower, self.upper = boundary.T
        assert np.all(self.lower < self.upper), boundary
        self.distribution_index = np.float32(config.getfloat('crossover_sbx', 'distribution_index'))
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        if self.random.random() >= self.prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        assert len(parent1.shape) == 1, parent1.shape
        assert parent1.shape == parent2.shape, (parent1.shape, parent2.shape)
        return np.vectorize(lambda parent1, parent2, lower, upper: self.crossover(parent1, parent2, lower, upper))(parent1, parent2, self.lower, self.upper)

    def crossover(self, parent1, parent2, lower, upper, dtype=np.float32):
        assert lower <= parent1 <= upper, (lower, parent1, upper)
        assert lower <= parent2 <= upper, (lower, parent2, upper)
        distance = np.fabs(parent1 - parent2)
        if distance == 0:
            return parent1, parent2
        # Calculate the lower and upper spread factor
        spread_factor_lower = 1 + 2 * (min(parent1, parent2) - lower) / distance
        spread_factor_upper = 1 + 2 * (upper - max(parent1, parent2)) / distance
        assert spread_factor_lower >= 0, spread_factor_lower
        assert spread_factor_upper >= 0, spread_factor_upper
        # Amplify the probability distribution
        amplification_factor_lower = amplification_factor(spread_factor_lower, self.distribution_index)
        amplification_factor_upper = amplification_factor(spread_factor_upper, self.distribution_index)
        # Generate the lower and upper instance of the spread factor
        u = self.random.random()
        spread_factor1 = spread_factor_instance(amplification_factor_lower, self.distribution_index, u)
        spread_factor2 = spread_factor_instance(amplification_factor_upper, self.distribution_index, u)
        # Produce the child
        middle = (parent1 + parent2) / 2
        half_distance = distance / 2
        child1 = middle - spread_factor1 * half_distance
        child2 = middle + spread_factor2 * half_distance
        child1 = np.clip(child1, lower, upper)
        child2 = np.clip(child2, lower, upper)
        child1, child2 = dtype(child1), dtype(child2)
        if self.random.random() < 0.5:
            return child1, child2
        else:
            return child2, child1


class Test(unittest.TestCase):
    def setUp(self):
        self.config = configparser.ConfigParser()
        wuji.config.load(self.config, os.sep.join(__file__.split(os.sep)[:-5] + ['config.ini']))
        evaluator = types.SimpleNamespace()
        evaluator.context = dict(encoding=dict(real=dict(boundary=np.array([(-3, 5), (-1, 4)]))))
        self.crossover = Crossover(self.config, evaluator)
        self.parent1 = np.array([-2, 3], np.float32)
        self.parent2 = np.array([3, 0], np.float32)

    def test_same(self):
        child1, child2 = self.crossover(self.parent1, self.parent1)
        np.testing.assert_array_almost_equal(self.parent1, child1)
        np.testing.assert_array_almost_equal(self.parent1, child2)

    def test_different(self):
        self.crossover(self.parent1, self.parent2)


if __name__ == '__main__':
    unittest.main()
