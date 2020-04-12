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
import copy
import types
import functools
import contextlib
import itertools
import pickle
import configparser
import logging
import traceback

import numpy as np
import torch.autograd

import wuji
import wuji.evaluator
from wuji.ea.crossover import Crossover
from wuji.ea.mutation import Mutation


class Runner(object):
    @staticmethod
    def ray_resources(config):
        evaluator = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, config.get('ea', 'evaluator').split('\t')))
        return evaluator.ray_resources(config)

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        try:
            seed = kwargs['seed']
            wuji.random.seed(seed, prefix=f'runner[{kwargs["index"]}].seed={seed}: ')
        except KeyError:
            pass
        kwargs['root_log'] = os.path.join(kwargs['root'], 'train', str(kwargs['index']))
        torch.set_num_threads(1)
        evaluator = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, config.get('ea', 'evaluator').split('\t')))
        self.evaluator = evaluator(config, **kwargs)
        try:
            crossover = {key: functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, value.split('\t'))) for key, value in self.config.items('crossover')}
            self._crossover = Crossover(config, self.evaluator, crossover)
            self._choose = self._crossover.choose
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            self._crossover = lambda population: population
            self._choose = 1
            logging.warning('crossover disabled')
        try:
            mutation = {key: functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, value.split('\t'))) for key, value in self.config.items('mutation')}
            self._mutation = Mutation(config, self.evaluator, mutation)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            self._mutation = lambda individual: individual
            logging.warning('mutation disabled')
        self.stopper = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, config.get('runner', 'stopper').split('\t')))

    def close(self):
        return self.evaluator.close()

    def __len__(self):
        return len(self.evaluator)

    def get_context(self):
        return pickle.dumps(self.evaluator.context)

    def digest(self, decision):
        return wuji.digest(decision, self.evaluator.context['encoding'])

    def initialize(self):
        decision = self.evaluator.initialize()
        individual = dict(
            decision=decision,
            digest=self.digest(decision),
            age=0,
        )
        return [individual]

    def choose(self):
        return self._choose

    def crossover(self, population):
        return self._crossover(population)

    def mutate(self, population):
        return [self._mutation(individual) for individual in population]

    def variation(self, population):
        for name in self.config.get('runner', 'variation').split():
            func = getattr(self, name)
            population = func(population)
        return population

    def evaluate(self, population):
        for individual in population:
            age = individual.get('age', 0)
            self.evaluator.cost = age
            decision = individual['decision']
            self.evaluator.set(decision)
            individual['evaluate'] = {}
            with contextlib.closing(wuji.Interval()) as interval:
                individual['result'] = self.evaluator.evaluate()
            individual['evaluate']['duration'] = interval.get()
            individual['cost'] = self.evaluator.cost - age
            individual['age'] = self.evaluator.cost
            individual['digest'] = self.digest(decision)
        return population

    def evolve(self, ancestor):
        offspring = self.variation(ancestor)
        for parent, child in zip(itertools.cycle(ancestor), offspring):
            child['age'] = parent['age']
        return self.evaluate(offspring)

    def train(self, population):
        for individual in population:
            age = individual.get('age', 0)
            self.evaluator.cost = age
            decision = individual['decision']
            self.evaluator.set(decision)
            individual['train'] = dict(iteration=0, duration=0)
            individual['evaluate'] = dict(duration=0)
            try:
                with contextlib.closing(self.evaluator.training()), torch.autograd.detect_anomaly():
                    with contextlib.closing(self.stopper(self.evaluator)) as stopper:
                        try:
                            with contextlib.closing(wuji.Interval()) as interval:
                                while True:
                                    outcome = self.evaluator.train()
                                    individual['train']['iteration'] += 1
                                    try:
                                        with contextlib.closing(wuji.Interval()) as _interval:
                                            if stopper(outcome):
                                                break
                                    finally:
                                        individual['evaluate']['duration'] += _interval.get()
                        finally:
                            individual['train']['duration'] += interval.get()
                # decision
                for key, value in copy.deepcopy(self.evaluator.get()).items():
                    individual['decision'][key] = value
                # evaluate
                try:
                    assert isinstance(stopper.result, dict), stopper.result
                    individual['result'] = stopper.result
                except AttributeError:
                    with contextlib.closing(wuji.Interval()) as interval:
                        individual['result'] = self.evaluator.evaluate()
                    individual['evaluate']['duration'] += interval.get()
            except:
                traceback.print_exc()
            finally:
                individual['cost'] = self.evaluator.cost - age
                individual['age'] = self.evaluator.cost
        return population

    def learn(self, ancestor):
        offspring = self.variation(ancestor)
        for parent, child in zip(itertools.cycle(ancestor), offspring):
            child['age'] = parent['age']
        offspring = self.train(offspring)
        for individual in offspring:
            individual['digest'] = self.digest(individual['decision'])
        return offspring

    def set_opponents_train(self, opponents, *args, **kwargs):
        return self.evaluator.set_opponents_train(opponents, *args, **kwargs)

    def set_opponents_eval(self, opponents):
        return self.evaluator.set_opponents_eval(opponents)
