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

import numpy as np

import wuji.record


def population(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def fetch(population, tag):
        keys = {key for individual in population if tag in individual for key in individual[tag]}
        return {'/'.join([name, tag, key]): np.array([individual[tag][key] for individual in population if tag in individual]) for key in keys}

    def get(population):
        if population:
            return {
                **{'/'.join([name, key]): [individual[key] for individual in population] for key in ('age', 'cost')},
                **{'/'.join([name, 'result', key]): np.array([individual['result'][key] for individual in population]) for key, value in population[0]['result'].items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))},
                **fetch(population, 'evaluate'),
                **fetch(population, 'train'),
            }
        else:
            return {}

    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'histogram_dict'), lambda: wuji.record.HistogramDict(self.cost, **get(getattr(self, name))), self.config.getboolean('record', '_histogram_dict'))
            return recorder
    return Optimizer


def offspring(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def fetch(population, tag):
        keys = {key for individual in population if tag in individual for key in individual[tag]}
        return {'/'.join([name, tag, key]): np.array([individual[tag][key] for individual in population if tag in individual]) for key in keys}

    def get(population):
        if population:
            return {
                **{'/'.join([name, key]): [individual[key] for individual in population] for key in ('age', 'cost')},
                **{'/'.join([name, 'result', key]): np.array([individual['result'][key] for individual in population]) for key, value in population[0]['result'].items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))},
                **fetch(population, 'evaluate'),
                **fetch(population, 'train'),
            }
        else:
            return {}

    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'histogram_dict'), lambda: wuji.record.HistogramDict(self.cost, **get(getattr(self, name))), self.config.getboolean('record', '_histogram_dict'))
            return recorder
    return Optimizer


def elite(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def get(individual):
        return {
            **{'/'.join([name, key]): individual[key] for key in ('age', 'cost')},
            **{'/'.join([name, 'result', key]): value for key, value in individual['result'].items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))},
            **{'/'.join([name, tag, key]): value for tag in ('evaluate', 'train') if tag in individual for key, value in individual[tag].items()},
        }

    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **get(getattr(self, name))))
            return recorder
    return Optimizer
