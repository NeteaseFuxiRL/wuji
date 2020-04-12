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
import collections
import random
import functools

import torch

import wuji

from .. import ENCODING, Crossover as _Crossover


class Split(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        encoding = evaluator.context['encoding'][ENCODING]
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][evaluator.get_kind()]['kwargs'])
        self.group = model.group()
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        if self.random.random() >= self.prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        point = self.random.choice(self.group + [None])
        child1 = []
        child2 = []
        for group in self.group:
            if group == point:
                parent1, parent2 = parent2, parent1
            for key in group:
                child1.append((key, parent1[key]))
                child2.append((key, parent2[key]))
        return collections.OrderedDict(child1), collections.OrderedDict(child2)


class Split1(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        encoding = evaluator.context['encoding'][ENCODING]
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][evaluator.get_kind()]['kwargs'])
        self.group = model.group()
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        if self.random.random() >= self.prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        point = self.random.choice(self.group + [None])
        parent = parent1
        child = []
        for group in self.group:
            if group == point:
                parent = parent2
            for key in group:
                child.append((key, parent[key]))
        return collections.OrderedDict(child),


class SplitLayerAny(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        encoding = evaluator.context['encoding'][ENCODING]
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][evaluator.get_kind()]['kwargs'])
        self.group = model.group()
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        if self.random.random() >= self.prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        point = self.random.choice(self.group)
        child1 = []
        child2 = []
        for group in self.group:
            if group == point:
                _key = group[0]
                channels = parent1[_key].size(0)
                i = self.random.randint(0, channels)
                for key in group:
                    assert parent1[key].size() == parent2[key].size()
                    if parent1[key].size():
                        assert parent1[key].size(0) == channels
                        child1.append((key, torch.cat([parent1[key][:i], parent2[key][i:]])))
                        child2.append((key, torch.cat([parent2[key][:i], parent1[key][i:]])))
                parent1, parent2 = parent2, parent1
            else:
                for key in group:
                    child1.append((key, parent1[key]))
                    child2.append((key, parent2[key]))
        return collections.OrderedDict(child1), collections.OrderedDict(child2)


class SplitLayerAny1(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        encoding = evaluator.context['encoding'][ENCODING]
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][evaluator.get_kind()]['kwargs'])
        self.group = model.group()
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        if self.random.random() >= self.prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        point = self.random.choice(self.group)
        parent = parent1
        child = []
        for group in self.group:
            if group == point:
                _key = group[0]
                channels = parent1[_key].size(0)
                i = self.random.randint(0, channels)
                for key in group:
                    assert parent1[key].size() == parent2[key].size()
                    if parent1[key].size():
                        assert parent1[key].size(0) == channels
                        layer = torch.cat([parent1[key][:i], parent2[key][i:]])
                        assert layer.size() == parent1[key].size()
                        child.append((key, layer))
                parent = parent2
            else:
                for key in group:
                    child.append((key, parent[key]))
        return collections.OrderedDict(child),


class SplitLayerAll(_Crossover):
    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.prob = config.getfloat('crossover_prob', ENCODING)
        encoding = evaluator.context['encoding'][ENCODING]
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][evaluator.get_kind()]['kwargs'])
        self.group = model.group()
        self.random = random.Random()

    def __call__(self, parent1, parent2):
        if self.random.random() >= self.prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        child1 = []
        child2 = []
        for group in self.group:
            channels = parent1[group[0]].size(0)
            i = self.random.randint(0, channels)
            for key in group:
                assert parent1[key].size() == parent2[key].size()
                if parent1[key].size():
                    assert parent1[key].size(0) == channels
                    child1.append((key, torch.cat([parent1[key][:i], parent2[key][i:]])))
                    child2.append((key, torch.cat([parent2[key][:i], parent1[key][i:]])))
        return collections.OrderedDict(child1), collections.OrderedDict(child2)
