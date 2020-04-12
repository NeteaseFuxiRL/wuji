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

import collections
import functools
import random

import numpy as np

import wuji.record


def blob(optimizer):
    def record(self):
        agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.context['encoding']['blob']['agent']['eval']))
        individual = random.choice(self.population)
        wuji.record.HistogramDict(self.cost, **collections.OrderedDict([('blob/' + key, var) for key, var in zip(agent.keys(individual['decision']['blob']), agent.values(individual['decision']['blob']))]))

    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'histogram_dict'), lambda: record(self))
            return recorder
    return Optimizer


def hparam(optimizer):
    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'histogram_dict'), lambda: wuji.record.HistogramDict(self.cost, **{f'hparam/population/{key}/{name}': np.copy([individual['decision'][key][i] for individual in self.population]).T for key, encoding in self.context['encoding'].items() if 'header' in self.context['encoding'][key] for i, name in enumerate(self.context['encoding'][key]['header'])}))
            return recorder
    return Optimizer
