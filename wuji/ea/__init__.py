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

import copy

import numpy as np
import ray


@ray.remote
def put_dict(individual, key, value):
    individual[key] = value
    return individual


@ray.remote
def get_dict(individual, key):
    return individual[key]


@ray.remote
def select_dict(individual, *args):
    return {key: individual[key] for key in args}


@ray.remote
def duplicate(individual):
    return copy.deepcopy(individual)


def distance_matrix(points, distance=lambda point1, point2: np.sqrt(np.sum((point1 - point2) ** 2)), dtype=None):
    try:
        if dtype is None:
            dtype = points[0].dtype
    except AttributeError:
        pass
    matrix = np.zeros([len(points), len(points)], dtype)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            matrix[i, j] = matrix[j, i] = distance(points[i], points[j])
    return matrix
