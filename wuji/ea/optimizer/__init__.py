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

import os
import itertools
import random
import operator
import pickle
import types
import logging
import functools
import configparser
import numbers
import warnings

import numpy as np
import torch
import ray
import tqdm

import wuji
import wuji.ea.mating
import wuji.ea.selection


def ensure_encoding(population, encoding, runner):
    _population = list(itertools.chain(*ray.get([r.initialize.remote() for r, _ in zip(itertools.cycle(runner), population)])))
    for individual, _individual in zip(population, _population):
        decision, _decision = individual['decision'], _individual['decision']
        changed = False
        for key in encoding:
            _chromo = _decision[key]
            try:
                assert key in decision
                chromo = decision[key]
                if type(_chromo) is np.ndarray:
                    assert chromo.shape == _chromo.shape
                    lower, upper = encoding[key]['boundary'].T
                    assert np.all(lower <= chromo)
                    assert np.all(chromo <= upper)
            except (AssertionError, KeyError):
                individual['decision'][key] = _chromo
                changed = True
        if changed:
            individual['digest'] = wuji.digest(decision, encoding)
    return population


class Optimizer(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        if 'root' not in kwargs:
            kwargs['root'] = os.path.expanduser(os.path.expandvars(config.get('model', 'root')))
        if 'log' not in kwargs:
            kwargs['log'] = wuji.config.digest(config)
        if 'root_log' not in kwargs:
            kwargs['root_log'] = os.path.join(kwargs['root'], 'log', kwargs['log'])
        self.runner = self.create_runner(**kwargs)
        self.parallel = sum(ray.get([runner.__len__.remote() for runner in self.runner]))
        self.context = pickle.loads(ray.get(self.runner[0].get_context.remote()))
        self.cost = 0
        self.iteration = 0

    def close(self):
        return ray.get([runner.close.remote() for runner in self.runner])

    def __len__(self):
        return self.parallel

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key in {'context', 'cost', 'iteration', 'population'}}

    def __setstate__(self, state):
        state.pop('context', None)
        for key, value in state.items():
            setattr(self, key, value)

    def create_runner(self, **kwargs):
        runner = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.config.get('ea', 'runner').split('\t')))
        runner = ray.remote(**runner.ray_resources(self.config))(runner)
        config = pickle.dumps(self.config)
        try:
            seed = kwargs.pop('seed')
        except KeyError:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        def make(index):
            return runner.remote(config, {**kwargs, **dict(index=index, seed=seed + index)})
        return [make(index) for index in range(kwargs['parallel'])]

    def digest(self, decision):
        return wuji.digest(decision, self.context['encoding'])

    def evaluate(self, population, **kwargs):
        population = list(itertools.chain(*ray.get(wuji.ray.submit(self.runner, iter(tqdm.tqdm([functools.partial(lambda runner, individual: runner.evaluate.remote([individual]), individual=individual) for individual in population], **kwargs))))))
        self.cost += sum(map(operator.itemgetter('cost'), population))
        return population

    def ensure_encoding(self, population):
        return ensure_encoding(population, self.context['encoding'], self.runner)

    def transfer(self, path):
        if os.path.isdir(path):
            path, _ = wuji.file.load(path)
        data = torch.load(path, map_location=lambda storage, loc: storage)
        try:
            return data['population']
        except KeyError:
            return [dict(decision=data['decision'])]


class EA(Optimizer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        schedule = [s.split(':') for s in config.get('ea', 'schedule').split('\t')]
        self.schedule = itertools.cycle(itertools.chain(*[itertools.repeat(name, eval(repeat)) for name, repeat in schedule]))
        self.initialize()
        self.population = self.evaluate(self.population, desc='evaluate population')
        # selector
        self.mating = self.create_mating()
        selection = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, config.get('ea', 'selection').split('\t')))
        self.selection = selection(self.config)
        self._population = config.getint('ea', 'population')

    def close(self):
        self.mating.close()
        self.selection.close()
        return super().close()

    def initialize(self):
        try:
            self.initialize_load()
        except (FileNotFoundError, ValueError):
            self.initialize_random()
        self.population += list(itertools.chain(*[self.transfer(os.path.expanduser(os.path.expandvars(path))) for path in self.kwargs['transfer']]))
        self.ensure_encoding(self.population)

    def initialize_load(self):
        path, self.cost = wuji.file.load(self.kwargs['root'])
        logging.info('load ' + path)
        self.__setstate__(torch.load(path, map_location=lambda storage, loc: storage))

    def initialize_random(self):
        self.cost = 0
        self.population = list(itertools.chain(*ray.get(wuji.ray.submit(self.runner, iter(tqdm.tqdm([lambda runner: runner.initialize.remote() for _ in range(self._population)], desc='random initialize'))))))
        assert len(set([individual['digest'] for individual in self.population])) == len(self.population), '\n'.join([individual['digest'] for individual in self.population])

    def create_mating(self):
        mating = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.config.get('ea', 'mating').split('\t')))
        return mating(self.config, self.population)

    def variation(self, runner):
        choose = ray.get(runner.choose.remote())
        ancestor = self.mating(choose)
        assert len(ancestor) == choose, (len(self.population), choose)
        command = next(self.schedule)
        return getattr(runner, command).remote(ray.put(ancestor))

    def select(self, offspring, size):
        return self.selection(offspring, size)

    def mix_population(self):
        try:
            population = self.population[:self.config.getint('ea', 'mix_population')]
        except (configparser.NoOptionError, ValueError):
            population = self.population
        return population


class Async(EA):
    def __init__(self, config, **kwargs):
        self._population = config.getint('ea', 'population')
        try:
            self._offspring = eval(config.get('ea', 'offspring'))
            assert isinstance(self._offspring, numbers.Integral), self._offspring
        except (configparser.NoOptionError, SyntaxError):
            self._offspring = self._population
            logging.warning(f'the offspring size is not configured, use the size of the population={self._offspring}')
        logging.info(f'population size={self._population}, offspring size={self._offspring}')
        super().__init__(config, **kwargs)
        if self._population < len(self.runner):
            warnings.warn(f'population size ({self._population}) is less than then number of runners ({len(self.runner)})')
        self.task = [self.variation(runner) for runner in self.runner]

    def idle_runner(self):
        (ready,), _ = ray.wait(self.task)
        index = self.task.index(ready)
        return self.runner[index]

    def spawn(self):
        (ready,), _ = ray.wait(self.task)
        offspring = ray.get(ready)
        index = self.task.index(ready)
        self.task[index] = self.variation(self.runner[index])
        return offspring

    def breeding(self, size, **kwargs):
        population = []
        while len(population) < size:
            offspring = self.spawn()
            cost = sum(map(operator.itemgetter('cost'), offspring))
            try:
                kwargs['pbar'].update(cost)
            except KeyError:
                pass
            self.cost += cost
            population += offspring
        return population

    def __call__(self, **kwargs):
        cost = self.cost
        self.offspring = self.breeding(self._offspring, **kwargs)
        if self.population:
            individual = random.choice(self.population)
            assert self.digest(individual['decision']) == individual['digest']
        mixed = self.mix_population() + self.offspring
        assert len(mixed) >= self._population, (len(mixed), self._population)
        self.population = self.select(mixed, self._population)
        assert len(self.population) == self._population, (len(self.population), self._population)
        self.mating = self.create_mating()
        self.iteration += 1
        return types.SimpleNamespace(cost=self.cost - cost)
