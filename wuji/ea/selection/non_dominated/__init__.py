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

import itertools

import numpy as np

import wuji.record
import wuji.ea.pareto
from wuji.ea.non_dominated import extract
from wuji.ea.pareto import *


def select(population, size, dominate):
    layers = []
    selected = 0
    while selected < size:
        non_dominated = extract(population, dominate)
        selected += len(non_dominated)
        layers.append(non_dominated)
    return layers


class Selection(object):
    def __init__(self, config):
        self.config = config
        self.dominate = eval('lambda individual1, individual2: ' + config.get('non_dominated', 'dominate'))

    def __call__(self, population, size):
        self.layers = select(population, size, self.dominate)
        non_critical = self.layers[:-1]
        for layer in non_critical:
            self.non_critical(layer)
        non_critical = list(itertools.chain(*non_critical))
        assert len(non_critical) < size, (len(non_critical), size)
        _size = size - len(non_critical)
        assert len(self.layers[-1]) >= _size, (len(self.layers[-1]), _size)
        if len(self.layers[-1]) > _size:
            self.critical = self.select_critical(self.layers[-1], _size)
        else:
            self.critical = self.layers[-1]
        assert len(self.critical) == _size, (len(self.critical), _size)
        return non_critical + self.critical

    def non_critical(self, population):
        raise NotImplementedError()

    def select_critical(self, population, size):
        raise NotImplementedError()

    def record(self, optimizer):
        optimizer.recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(optimizer.cost, **{
            'non_dominated/layers': len(self.layers),
            'non_dominated/critical': len(self.critical),
        }))


def test():
    for n, dimension in [(200, 2), (300, 3)]:
        require = n / 2
        population = [np.random.rand(dimension) for _ in range(n)]
        layers = select(population, require, dominate_min)
        assert(len(layers) > 0)
        non_critical = sum([len(layer) for layer in layers[:-1]])
        candidate = non_critical + len(layers[-1])
        assert(non_critical < require <= candidate)
        assert(candidate <= n)
        assert(candidate + len(population) == n)
