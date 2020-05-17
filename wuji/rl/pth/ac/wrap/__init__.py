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
import logging

import numpy as np
import torch
import torch.nn.functional as F

from . import problem, loss
from wuji.rl.pth import ac

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def attr(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_critic = getattr(F, self.config.get(NAME, 'loss_critic'))
            self.weight_loss = torch.FloatTensor(np.array([self.config.getfloat(NAME + '_weight_loss', key) for key in ac.Losses._fields]))
    return RL


def gae(rl):
    from .. import gae
    name = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, name)
            setattr(self, name, self.config.getfloat(NAME, name))
            if getattr(self, name) <= 0:
                self.value = super().value
                logging.warning(f'{name.upper()} disabled')

        def value(self, **kwargs):
            _baseline = torch.cat([kwargs['baseline'], kwargs['terminal'].view(-1)])
            return gae(kwargs['reward'], kwargs['discount'], _baseline, getattr(self, name))
    return RL
