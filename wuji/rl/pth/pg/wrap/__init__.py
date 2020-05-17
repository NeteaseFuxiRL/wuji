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

from . import problem, hook

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def attr(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.discount = self.config.getfloat('rl', 'discount')
            self.norm = eval('lambda reward: ' + self.config.get(NAME, 'norm'))
    return RL


def agent(rl):
    class RL(rl):
        def set_prob_min(self, prob_min):
            return self.agent.set_prob_min(prob_min)

        def get_prob_min(self):
            return self.agent.prob_min
    return RL
