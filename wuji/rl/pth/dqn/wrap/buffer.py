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
import random
import collections

import numpy as np

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def fifo(rl):
    class Buffer(collections.deque):
        def sample(self, size):
            return random.sample(self, size)

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.buffer = Buffer(maxlen=self.config.getint(NAME, 'capacity'))
    return RL


def diverse(rl):
    from ..buffer import Diverse as Buffer
    name = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            choose = self.config.getint('_'.join([NAME, name]), 'choose')
            space = np.insert(self.problem.context['state_space'], -1, [0, self.problem.context['encoding']['blob']['init'][self.kind]['kwargs']['outputs'] - 1], axis=0)
            lower, upper = np.moveaxis(space, -1, 0)
            range = upper - lower + np.finfo(space.dtype).eps
            self.buffer = Buffer(self.config.getint(NAME, 'capacity'), choose, lower, range)
    return RL
