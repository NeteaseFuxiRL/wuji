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
import operator

import numpy as np
import dmaps


class Diverse(list):
    def __init__(self, capacity, choose, lower, range):
        super().__init__()
        self.capacity = capacity
        self.choose = choose
        self.lower = lower
        self.range = range

    def append(self, exp):
        point = np.insert(exp.state.cpu().numpy().flatten(), -1, exp.action)
        point = (point - self.lower) / self.range
        try:
            super().append((point, exp))
        finally:
            if len(self) > self.capacity:
                choose = random.sample(range(len(self)), self.choose)
                dist = dmaps.DistanceMatrix([self[i][0] for i in choose])
                dist.compute(metric=dmaps.metrics.euclidean)
                matrix = dist.get_distances()
                # matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist([self[i][0] for i in choose]))
                for i in range(len(matrix)):
                    matrix[i, i] = np.finfo(matrix.dtype).max
                i, j = np.unravel_index(np.argmin(matrix), matrix.shape)
                self.pop(choose[i])

    def __iadd__(self, trajectory):
        for exp in trajectory:
            self.append(exp)
        return self

    def sample(self, size):
        return list(map(operator.itemgetter(1), random.sample(self, size)))
