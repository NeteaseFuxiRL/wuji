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

import torch

from .. import agent

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def context(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context['encoding']['blob']['module'] = self.config.get('model', 'module').split() + self.config.get('model', 'init').split()
            self.context['encoding']['blob']['agent'] = dict(
                train=['.'.join([agent.__name__, 'Train'])] + self.config.get(NAME, 'agent_train').split(),
                eval=['.'.join([agent.__name__, 'Eval'])] + self.config.get(NAME, 'agent_eval').split(),
            )
    return RL


def batch_size(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, name)
            setattr(self, name, self.config.getint('train', name))

        def set_batch_size(self, size):
            setattr(self, name, size)

        def get_batch_size(self):
            return getattr(self, name)

        def __next__(self):
            cost, tensors, result = super().__next__()
            while len(tensors[0]) < getattr(self, name):
                _cost, _tensors, _result = super().__next__()
                cost += _cost
                tensors = tuple(torch.cat([tensor, _tensor]) for tensor, _tensor in zip(tensors, _tensors))
                if _result:
                    result = _result
            return cost, tensors, result
    return RL
