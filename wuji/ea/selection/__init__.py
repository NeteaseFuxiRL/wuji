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
import copy

import numpy as np


def partition(predicate, values):
    results = ([], [])
    for item in values:
        results[predicate(item)].append(item)
    return results


def front(population, size):
    return population[:size]


def truncate(population, size, key=lambda individual: individual['result']['fitness']):
    return sorted(population, key=key, reverse=True)[:size]


class Random(object):
    def __init__(self, config):
        self.config = config

    def close(self):
        pass

    def __call__(self, population, size):
        return random.sample(population, size)


class Truncation(object):
    def __init__(self, config):
        self.config = config

    def close(self):
        pass

    def __call__(self, population, size):
        return truncate(population, size)


def tournament(population, competitors, compete=lambda *competitors: max(competitors, key=lambda item: item[1]['fitness'])):
    competitors = random.sample(list(enumerate(population)), competitors)
    index, individual = compete(*competitors)
    return index, individual


class Tournament(object):
    def __init__(self, config):
        self.config = config

    def close(self):
        pass

    def __call__(self, population, size):
        return tournament(population, size)


def _roulette_wheel(population, total, key):
    end = random.uniform(0, total)
    seek = 0
    for i, individual in enumerate(population):
        seek += individual[key]
        if seek >= end:
            return i
    return -1


def roulette_wheel(population, size, key='fitness', elites=1):
    assert elites < size
    assert size < len(population)
    selected = []
    for _ in range(elites):
        i = np.argmax([individual[key] for individual in population])
        elite = population.pop(i)
        selected.append(elite)
    lower = min(individual[key] for individual in population)
    _key = '_' + key
    for individual in population:
        individual[_key] = individual[key] - lower
    for _ in range(size - elites):
        total = sum(individual[_key] for individual in population)
        if total > 0:
            i = _roulette_wheel(population, total, _key)
            elite = population.pop(i)
            selected.append(elite)
        else:
            return selected + population[:size - len(selected)]
    return selected


class RouletteWheel(object):
    def __init__(self, config):
        self.config = config

    def close(self):
        pass

    def __call__(self, population, size):
        return roulette_wheel(population, size)


class StochasticTournamentSelection(object):
    def __init__(self, config):
        self.config = config
        self.compare = eval('lambda individual: ' + config.get('tournament', 'compare'))
        self.probability = config.getfloat('tournament', 'probability')
        assert self.probability > 0.5, self.probability

    def compete(self, competitor1, competitor2):
        index1, parent1 = competitor1
        index2, parent2 = competitor2
        if parent1['result'][self.compare] > parent2['result'][self.compare]:
            if random.random() < self.probability:
                return competitor1
            else:
                return competitor2
        else:
            if random.random() < self.probability:
                return competitor2
            else:
                return competitor1

    def __call__(self, population, size):
        population = copy.copy(population)
        selected = []
        while len(selected) < size:
            index, individual = tournament(population, 2, self.compete)
            population.pop(index)
            selected.append(individual)
        return selected
