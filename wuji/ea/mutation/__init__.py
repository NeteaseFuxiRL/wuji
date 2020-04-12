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

    def __call__(self, individual):
        return individual


class Mutation(object):
    def __init__(self, config, evaluator, mutation):
        self.config = config
        mutation = {key: mutation[key] for key in evaluator.context['encoding']}
        self.mutation = {key: mutation(config) if mutation.__init__.__code__.co_argcount == 2 else mutation(config, evaluator) for key, mutation in mutation.items()}

    def __call__(self, individual):
        name = {key: mutation.__class__.__name__ for key, mutation in self.mutation.items()}
        for key in individual['decision'].keys():
            try:
                mutation = self.mutation[key]
            except KeyError:
                mutation = lambda decision, **kwargs: decision
            individual['decision'][key] = mutation(individual['decision'][key], **individual.get('result', {}))
        individual['mutation'] = name
        return individual


class RandomChoice(object):
    def __init__(self, config, evaluator, *args):
        self.mutation = [arg(config, evaluator) for arg in args]
        self.random = random.Random()

    def __call__(self, *args, **kwargs):
        return self.random.choice(self.mutation)(*args, **kwargs)
