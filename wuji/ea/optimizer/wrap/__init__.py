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


def age(optimizer):
    def make(individual):
        individual['result']['objective'] = [(objective, individual['age']) for objective in individual['result']['objective']]
        return individual

    class Optimizer(optimizer):
        def evaluate(self, *args, **kwargs):
            return list(map(make, super().evaluate(*args, **kwargs)))

        def breeding(self, *args, **kwargs):
            return list(map(make, super().breeding(*args, **kwargs)))
    return Optimizer


def elite(optimizer):
    class Optimizer(optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.elite = copy.deepcopy(max(self.population, key=lambda individual: individual['result']['fitness']))

        def __call__(self, *args, **kwargs):
            ret = super().__call__(*args, **kwargs)
            candidate = max(self.population, key=lambda individual: individual['result']['fitness'])
            if candidate['result']['fitness'] > self.elite['result']['fitness']:
                self.elite = copy.deepcopy(candidate)
            return ret
    return Optimizer
