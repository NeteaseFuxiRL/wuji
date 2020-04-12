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
import dmaps

from . import Selection as _Selection


def drop(front, size):
    dist = dmaps.DistanceMatrix(front)
    dist.compute(metric=dmaps.metrics.euclidean)
    matrix = dist.get_distances()
    # matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(front))
    for i in range(len(matrix)):
        matrix[i, i] = np.finfo(matrix.dtype).max
    indexes = []
    for _ in range(len(front) - size):
        i, j = np.unravel_index(np.argmin(matrix), matrix.shape)
        indexes.append(i)
        for j in range(len(matrix)):
            matrix[i, j] = np.finfo(matrix.dtype).max
            matrix[j, i] = np.finfo(matrix.dtype).max
    return set(indexes)


class Selection(_Selection):
    def non_critical(self, population):
        pass

    def select_critical(self, population, size):
        indexes = drop([individual['point'] for individual in population], size)
        return [individual for i, individual in enumerate(population) if i not in indexes]
