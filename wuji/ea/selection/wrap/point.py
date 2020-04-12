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


def extreme(selection):
    class Selection(selection):
        def __call__(self, population, size):
            point = eval('lambda result: ' + self.config.get('multi_objective', 'point'))
            assert all(map(np.isscalar, point(population[0]['result']))), point(population[0]['result'])
            num = len(point(population[0]['result']))
            assert num < size < len(population), (num, size, len(population))
            index = {max(enumerate(population), key=lambda item: point(item[1]['result'])[i])[0] for i in range(num)}
            selected = [individual for i, individual in enumerate(population) if i in index]
            return selected + super().__call__([individual for i, individual in enumerate(population) if i not in index], size - len(selected))
    return Selection
