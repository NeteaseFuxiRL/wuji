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
import configparser
import logging

import numpy as np
import torch

from wuji.problem.mdp import Truncator
from .. import agent

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def context(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context['encoding']['blob']['module'] += self.config.get('model', 'critic').split()
            self.context['encoding']['blob']['agent']['train'][0] = '.'.join([agent.__name__, 'Train'])
            self.context['encoding']['blob']['agent']['eval'][0] = '.'.join([agent.__name__, 'Eval'])
    return RL


def truncation(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, name)
            truncator = Truncator(self.problem, self.kind, self.get_agent, self.get_opponent_train_agent, self.context['length'])
            truncator.max = truncator.step
            setattr(self, name, truncator)
            try:
                self.set_truncation(self.config.getint(NAME, name))
            except configparser.NoOptionError:
                logging.warning(f'{name} disabled')

        def set_truncation(self, step):
            assert step > 0, step
            getattr(self, name).step = step

        def get_truncation(self):
            return getattr(self, name).step

        def rollout(self):
            truncator = getattr(self, name)
            if truncator.step >= truncator.max and truncator.done:
                return super().rollout()
            else:
                trajectory = next(truncator)
                with torch.no_grad():
                    inputs = tuple(self.agent.tensor(a.astype(np.float32), expand=0) for a in trajectory[-1]['state_']['inputs'])
                    logits, baseline = self.model(*inputs)
                return trajectory, baseline[0][0], truncator.controller.get_result() if truncator.done else {}

        def evaluate(self):
            truncator = getattr(self, name)
            while not truncator.done:
                next(truncator)
            return super().evaluate()

        def evaluate_map(self, *args, **kwargs):
            truncator = getattr(self, name)
            while not truncator.done:
                next(truncator)
            return super().evaluate_map(*args, **kwargs)
    return RL
