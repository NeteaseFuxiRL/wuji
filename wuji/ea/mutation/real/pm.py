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
@ARTICLE{,
  author = {Kalyanmoy Deb and Mayank Goyal},
  title = {A Combined Genetic Adaptive Search (GeneAS) for Engineering Design},
  journal = {Computer Science and Informatics},
  year = {1996},
  volume = {26},
  pages = {30-45},
  number = {4}
}
"""

import math
import numpy
import random
import configparser

import numpy as np

from . import ENCODING, Mutation as _Mutation


def calc_perturbance_factor_probability(perturbance_factor, distribution_index):
    return (distribution_index + 1) * numpy.power(1 - numpy.fabs(perturbance_factor), distribution_index) / 2


def calc_amplification_factor_lower(perturbance_factor_lower, distribution_index):
    assert numpy.all(-1 <= perturbance_factor_lower) and numpy.all(perturbance_factor_lower <= 0), perturbance_factor_lower
    assert distribution_index >= 0, distribution_index
    return 2 / (1 - numpy.power(1 + perturbance_factor_lower, distribution_index + 1))


def calc_amplification_factor_upper(perturbance_factor_upper, distribution_index):
    assert numpy.all(0 <= perturbance_factor_upper) and numpy.all(perturbance_factor_upper <= 1), perturbance_factor_upper
    assert distribution_index >= 0, distribution_index
    return 2 / (1 - numpy.power(1 - perturbance_factor_upper, distribution_index + 1))


def generate_perturbance_factor_instance_lower(amplification_factor_lower, distribution_index, u):
    assert amplification_factor_lower > 1, amplification_factor_lower
    return math.pow(2 * u + (1 - 2 * u) * (1 - 2 / amplification_factor_lower), 1. / (distribution_index + 1)) - 1


def generate_perturbance_factor_instance_upper(amplification_factor_upper, distribution_index, u):
    assert amplification_factor_upper > 1, amplification_factor_upper
    return 1 - math.pow(2 * (1 - u) + (2 * u - 1) * (1 - 2 / amplification_factor_upper), 1. / (distribution_index + 1))


def _calc_amplification_factor_lower(perturbance_factor_lower, distribution_index):
    assert -1 <= perturbance_factor_lower <= 0, perturbance_factor_lower
    assert distribution_index >= 0, distribution_index
    return math.pow(1 + perturbance_factor_lower, distribution_index + 1)


def _calc_amplification_factor_upper(perturbance_factor_upper, distribution_index):
    assert 0 <= perturbance_factor_upper <= 1, perturbance_factor_upper
    assert distribution_index >= 0, distribution_index
    return math.pow(1 - perturbance_factor_upper, distribution_index + 1)


def _generate_perturbance_factor_instance_lower(_amplification_factor_lower, distribution_index, u):
    return math.pow(2 * u + (1 - 2 * u) * _amplification_factor_lower, 1. / (distribution_index + 1)) - 1


def _generate_perturbance_factor_instance_upper(_amplification_factor_upper, distribution_index, u):
    return 1 - math.pow(2 * (1 - u) + (2 * u - 1) * _amplification_factor_upper, 1. / (distribution_index + 1))


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
        self.distribution_index = np.float32(config.getfloat('mutation_pm', 'distribution_index'))
        self.random = random.Random()

    def __call__(self, parent, **kwargs):
        assert len(parent.shape) == 1, parent.shape
        prob = self.prob(parent)
        return np.vectorize(lambda parent, lower, upper: self.mutate(parent, lower, upper) if self.random.random() < prob else parent)(parent, self.lower, self.upper)

    def mutate(self, parent, lower, upper, dtype=np.float32):
        assert lower <= parent <= upper, (lower, parent, upper)
        distance = upper - lower
        assert distance > 0, distance
        u = self.random.random()
        if u < 0.5:
            # Lower
            perturbance_factor_lower = (lower - parent) / distance
            _amplification_factor_lower = _calc_amplification_factor_lower(perturbance_factor_lower, self.distribution_index)
            perturbance_factor = _generate_perturbance_factor_instance_lower(_amplification_factor_lower, self.distribution_index, u)
            assert perturbance_factor <= 0, perturbance_factor
        else:
            # Upper
            perturbance_factor_upper = (upper - parent) / distance
            _amplification_factor_upper = _calc_amplification_factor_upper(perturbance_factor_upper, self.distribution_index)
            perturbance_factor = _generate_perturbance_factor_instance_upper(_amplification_factor_upper, self.distribution_index, u)
            assert 0 <= perturbance_factor, perturbance_factor
        child = parent + perturbance_factor * distance
        child = np.clip(child, lower, upper)
        return dtype(child)
