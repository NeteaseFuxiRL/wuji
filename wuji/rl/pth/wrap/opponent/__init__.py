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
import inspect
import types
import collections
import functools
import itertools
import hashlib

import wuji.problem.mdp

NAME = os.path.basename(os.path.dirname(__file__))


def make_agents(config, context, opponent, mode='eval'):
    agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, context['agent'][mode]))
    module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in context['module']))
    init = context['init']
    agents = {kind: agent(module(config, **init[kind]['kwargs'])) for kind in opponent}
    for kind, agent in agents.items():
        agent.model.set_blob(opponent[kind])
        agent.model.eval()
    return agents


def opponents_digest(opponents):
    return hashlib.md5(bytes(itertools.chain(*[bytes(itertools.chain(*[bytes(itertools.chain(*[value.cpu().numpy().tostring() for value in blob.values()])) for blob in opponent.values()])) for opponent in opponents]))).hexdigest()


def train(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, types.SimpleNamespace(opponent={}, agent={}))

        def set_opponent_train(self, opponent):
            assert isinstance(opponent, dict), type(opponent)
            attr = getattr(self, _name)
            attr.opponent = opponent
            attr.agent = make_agents(self.config, self.context['encoding']['blob'], opponent)

        def get_opponent_train(self):
            attr = getattr(self, _name)
            return attr.opponent

        def get_opponent_train_agent(self):
            attr = getattr(self, _name)
            return attr.agent
    return RL


def eval(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, types.SimpleNamespace(opponents=[{}], digest=''))

        def set_opponents_eval(self, opponents):
            assert isinstance(opponents, collections.abc.Iterable), type(opponents)
            assert opponents
            assert all(isinstance(opponent, dict) for opponent in opponents), [type(opponent) for opponent in opponents]
            attr = getattr(self, _name)
            attr.opponents = opponents
            if opponents[0]:
                attr.digest = opponents_digest(opponents)
            else:
                attr.digest = ''
            return attr.digest

        def get_opponents_eval(self):
            attr = getattr(self, _name)
            return attr.opponents

        def get_opponents_eval_digest(self):
            attr = getattr(self, _name)
            return attr.digest

        def evaluate_reduce(self, *args, **kwargs):
            result = super().evaluate_reduce(*args, **kwargs)
            digest = self.get_opponents_eval_digest()
            if digest:
                result['digest_opponents_eval'] = digest
            return result
    return RL
