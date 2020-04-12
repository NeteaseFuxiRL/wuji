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


class Useless(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, parent1, parent2):
        return parent1, parent2


class Crossover(object):
    def __init__(self, config, evaluator, crossover):
        self.config = config
        crossover = {key: crossover[key] for key in evaluator.context['encoding']}
        self.crossover = {key: crossover(config) if crossover.__init__.__code__.co_argcount == 2 else crossover(config, evaluator) for key, crossover in crossover.items()}
        self._choose = {key: crossover.__call__.__code__.co_argcount - 1 for key, crossover in self.crossover.items()}
        assert min(self._choose.values()) > 0 and max(self._choose.values()) > 1, self._choose
        self.choose = max(self._choose.values())
        self.name = {key: crossover.__class__.__name__ for key, crossover in self.crossover.items()}

    def __call__(self, ancestor):
        assert len(ancestor) == self.choose, len(ancestor)
        _offspring = {}
        for key, crossover in self.crossover.items():
            choose = self._choose[key]
            _ancestor = [individual['decision'][key] for individual in ancestor[:choose]]
            _offspring[key] = crossover(*_ancestor)
        offspring = [dict(decision={}, ancestors={individual['digest']: individual['result'] for individual in ancestor}, crossover=self.name) for _ in range(min(map(len, _offspring.values())))]
        for key, decisions in _offspring.items():
            assert len(decisions) >= len(offspring), (len(decisions), len(offspring))
            for decision, child in zip(decisions, offspring):
                child['decision'][key] = decision
        return offspring


class RandomChoice(object):
    def __init__(self, config, evaluator, *args):
        self.crossover = [arg(config, evaluator) for arg in args]
        self.random = random.Random()

    def __call__(self, *args, **kwargs):
        return self.random.choice(self.crossover)(*args, **kwargs)
