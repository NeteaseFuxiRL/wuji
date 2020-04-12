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


def tournament(population, competitors, compete=lambda *competitors: max(competitors, key=lambda item: item[1]['fitness']), random=random):
    competitors = random.sample(list(enumerate(population)), competitors)
    index, individual = compete(*competitors)
    return index, individual


def roulette_wheel(population, func, random=random):
    total = sum(map(func, population))
    end = random.uniform(0, total)
    seek = 0
    for i, individual in enumerate(population):
        seek += func(individual)
        if seek >= end:
            return i
    return -1


class Random(object):
    def __init__(self, config, population):
        self.config = config
        self.population = population
        self.random = random.Random()

    def close(self):
        pass

    def __call__(self, choose):
        return self.random.sample(self.population, choose)


class Tournament(object):
    def __init__(self, config, population):
        self.config = config
        self.population = population
        self.competitors = config.getint('tournament', 'competitors')
        assert self.competitors >= 2
        self.compare = eval('lambda individual: ' + config.get('tournament', 'compare'))
        self.random = random.Random()

    def close(self):
        pass

    def compete(self, *competitors):
        return max(competitors, key=lambda item: self.compare(item[1]))

    def __call__(self, choose):
        assert len(self.population) > self.competitors, (len(self.population), self.competitors)
        population = copy.copy(self.population)
        selected = []
        for _ in range(choose):
            index, individual = tournament(population, self.competitors, self.compete, random=self.random)
            population.pop(index)
            selected.append(individual)
        return selected
