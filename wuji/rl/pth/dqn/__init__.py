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

import numpy as np
import torch
import recordtype

from .. import wrap as _wrap
from . import wrap, agent

NAME = os.path.basename(os.path.dirname(__file__))
Outcome = recordtype.recordtype('Outcome', ['cost', 'loss', 'result'])


@wrap.reset_buffer
@wrap.truncation()
@_wrap.checkpoint
@_wrap.opponent.eval
@_wrap.opponent.train
@_wrap.evaluate
@_wrap.optimizer
@_wrap.agent
@_wrap.model
@wrap.problem.context
@_wrap.problem
@wrap.attr
@wrap.rollout(*'inputs inputs_ action reward done'.split())
class RL(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def close(self):
        pass

    def __len__(self):
        return 1

    def fill(self, size):
        cost = 0
        while True:
            trajectory, result = self.rollout()
            self.buffer += trajectory
            cost += sum(exp.get('cost', 1) for exp in trajectory)
            if cost >= size:
                break
        return cost, result

    def get_q_label(self, trajectory):
        reward = torch.FloatTensor(np.array([exp['reward'] for exp in trajectory]))
        discount = torch.FloatTensor(np.array([0 if exp['done'] else self.discount for exp in trajectory]))
        inputs_ = [exp['inputs_'] for exp in trajectory]
        inputs_ = tuple(map(lambda t: torch.cat(t), zip(*inputs_)))
        with torch.no_grad():
            q_ = self.model(*inputs_)
            q_max_, _ = q_.max(1)
        return reward + discount * q_max_

    def sample(self):
        trajectory = self.buffer.sample(self.batch_size)
        inputs = tuple(map(lambda t: torch.cat(t), zip(*[exp['inputs'] for exp in trajectory])))
        q_label = self.get_q_label(trajectory)
        action = torch.cat([exp['action'] for exp in trajectory])
        return inputs, q_label, action

    def __call__(self):
        with torch.no_grad():
            cost, result = self.fill(self.batch_size)
        inputs, q_label, action = self.sample()
        q = self.model(*inputs)
        q_action = q.gather(1, action.view(-1, 1)).view(-1)
        loss = self._loss(q_action, q_label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.agent.update()
        return Outcome(cost, loss, result)
