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
import collections

import numpy as np
import torch.cuda
import torch.nn.functional as F

from .. import wrap as _wrap
from . import wrap, agent

NAME = os.path.basename(os.path.dirname(__file__))
Outcome = collections.namedtuple('Outcome', ['cost', 'loss', 'result'])


@wrap.truncation()
@wrap.rollout(*'inputs inputs_ action reward done'.split())
@_wrap.problem
@_wrap.optimizer
@_wrap.agent
@_wrap.model
@_wrap.evaluate
class RL(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.batch_size = self.config.getint('train', 'batch_size')
        self.discount = config.getfloat('rl', 'discount')
        self._loss = getattr(F, config.get(NAME, 'loss'))

    def close(self):
        pass

    def update_context(self, context):
        context['encoding']['blob']['module'] = self.config.get('model', 'module').split() + self.config.get('model', 'init').split()
        context['encoding']['blob']['agent'] = dict(
            train=['.'.join([agent.__name__, 'Train'])] + self.config.get(NAME, 'agent_train').split(),
            eval=['.'.join([agent.__name__, 'Eval'])] + self.config.get(NAME, 'agent_eval').split(),
        )

    def fill(self, size):
        cost = 0
        while True:
            trajectory, result = self.rollout()
            self.buffer += trajectory
            cost += sum(exp.get('cost', 1) for exp in trajectory)
            if cost >= size:
                break
        return cost, result

    def value(self, trajectory):
        reward = torch.FloatTensor(np.array([exp['reward'] for exp in trajectory]))
        discount = torch.FloatTensor(np.array([0 if exp['done'] else self.discount for exp in trajectory]))
        inputs_ = [exp['inputs_'] for exp in trajectory]
        inputs_ = tuple(map(lambda t: torch.cat(t), zip(*inputs_)))
        with torch.no_grad():
            q_ = self.model(*inputs_)
            value_, _ = q_.max(1)
        return reward + discount * value_

    def sample(self):
        trajectory = self.buffer.sample(self.batch_size)
        inputs = tuple(map(lambda t: torch.cat(t), zip(*[exp['inputs'] for exp in trajectory])))
        value = self.value(trajectory)
        action = torch.cat([exp['action'] for exp in trajectory])
        return inputs, value, action

    def __call__(self):
        with torch.no_grad():
            cost, result = self.fill(self.batch_size)
        inputs, value_, action = self.sample()
        q = self.model(*inputs)
        value = q.gather(1, action.view(-1, 1)).view(-1)
        loss = self._loss(value, value_)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.agent.update()
        return Outcome(cost, loss, result)
