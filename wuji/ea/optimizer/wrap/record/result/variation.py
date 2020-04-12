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

import inspect
import numbers
import operator

import numpy as np

import wuji.record


def improve(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def get(population):
        if population:
            return {
                **{'/'.join([f'{name}_max', key]): np.array([individual['result'][key] - max(map(operator.itemgetter(key), individual['ancestors'].values())) for individual in population]) for key, value in population[0]['result'].items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))},
                **{'/'.join([f'{name}_min', key]): np.array([individual['result'][key] - min(map(operator.itemgetter(key), individual['ancestors'].values())) for individual in population]) for key, value in population[0]['result'].items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))},
            }
        else:
            return {}

    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'histogram_dict'), lambda: wuji.record.HistogramDict(self.cost, **get(self.offspring)), self.config.getboolean('record', '_histogram_dict'))
            return recorder
    return Optimizer
