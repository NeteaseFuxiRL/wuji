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
import functools

import numpy as np
import torch

import wuji
from . import buffer
from wuji.problem.mdp import Truncator

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def truncation(step=None):
    def decorate(rl):
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.truncator = Truncator(self.problem, self.kind, self.get_agent, lambda: {}, self.batch_size if step is None else step)

            def evaluate(self):
                while not self.truncator.done:
                    next(self.truncator)
                return super().evaluate()
        return RL
    return decorate


def rollout(*keys):
    keys = set(keys)

    def decorate(rl):
        class RL(rl):
            def rollout(self):
                trajectory = next(self.truncator)
                for exp, exp_ in zip(trajectory[:-1], trajectory[1:]):
                    exp['inputs_'] = exp_['inputs']
                exp = trajectory[-1]
                exp['inputs_'] = tuple(self.agent.tensor(a.astype(np.float32), expand=0) for a in exp['state_']['inputs'])
                return [{key: value for key, value in exp.items() if key in keys} for exp in trajectory], self.truncator.controller.get_result() if self.truncator.done else {}
        return RL
    return decorate


def double(rl):
    class RL(rl):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            assert not hasattr(self, 'model_')
            module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in self.context['encoding']['blob']['module']))
            self.model_ = module(self.config, **self.problem.context['encoding']['blob']['init'][self.kind]['kwargs'])
            self.model_.eval()
            self.model_.load_state_dict(self.model.state_dict())
            self.update = wuji.counter.Number(self.config.getint(NAME, 'update'))

        def set_blob(self, blob):
            super().set_blob(blob)
            self.model_.set_blob(blob)

        def value(self, trajectory):
            reward = torch.FloatTensor(np.array([exp['reward'] for exp in trajectory]))
            discount = torch.FloatTensor(np.array([0 if exp['done'] else self.discount for exp in trajectory]))
            inputs_ = [exp['inputs_'] for exp in trajectory]
            inputs_ = tuple(map(lambda t: torch.cat(t), zip(*inputs_)))
            with torch.no_grad():
                q_ = self.model_(*inputs_)
                value_, _ = q_.max(1)
            return reward + discount * value_

        def __call__(self):
            try:
                return super().__call__()
            finally:
                if self.update():
                    self.model_.load_state_dict(self.model.state_dict())
    return RL
