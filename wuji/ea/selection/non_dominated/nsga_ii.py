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
import collections

import numpy as np

from . import Selection as _Selection


Item = collections.namedtuple('Item', ['point', 'individual'])
NAME = os.path.basename(os.path.dirname(__file__))


def crowding_distance(population, point):
    for individual in population:
        individual['crowding_distance'] = 0
    items = [Item(point(individual['result']), individual) for individual in population]
    for i in range(len(items[0].point)):
        items.sort(key=lambda item: item.point[i])
        lower, upper = items[0].point[i], items[-1].point[i]
        dist = upper - lower
        assert dist >= 0, dist
        if dist > 0:
            items[0].individual['crowding_distance'] = np.inf
            items[-1].individual['crowding_distance'] = np.inf
            for item1, item, item2 in zip(items[:-2], items[1:-1], items[2:]):
                p1, p2 = item1.point[i], item2.point[i]
                assert p1 <= item.point[i] <= p2
                item.individual['crowding_distance'] += (p2 - p1) / dist
    return population


class Selection(_Selection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point = eval('lambda result: ' + self.config.get('multi_objective', 'point'))
        self.density = eval('lambda individual: ' + self.config.get('nsga_ii', 'density'))

    def non_critical(self, population):
        crowding_distance(population, self.point)

    def select_critical(self, population, size):
        crowding_distance(population, self.point)
        population.sort(key=self.density, reverse=True)
        return population[:size]
