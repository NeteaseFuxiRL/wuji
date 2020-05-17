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
import recordtype

import wuji
from .. import ac
from ..ac import Losses, Outcome as _Outcome

NAME = os.path.basename(os.path.dirname(__file__))
Outcome = recordtype.recordtype('Outcome', _Outcome._fields + ('gradient',))


class Actor(ac.RL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = kwargs['seed']
        wuji.random.seed(seed, prefix=f'{NAME.upper()}[{kwargs["index"]}].seed={seed}: ')

    def set_weight_loss(self, i, weight):
        self.weight_loss[i] = weight

    def get_weight_loss(self, i):
        return self.weight_loss[i]

    def gradient(self, blob, **kwargs):
        for key, (args, kwargs) in kwargs.items():
            getattr(self, key)(*args, **kwargs)
        self.set_blob(blob)
        outcome = self.backward()
        return Outcome(outcome.cost, outcome.loss.item(), Losses(*[loss.item() for loss in outcome.losses]), {key: value for key, value in outcome.result.items() if np.isscalar(value) and not key.startswith('_')}, [param.grad for param in self.model.parameters()])
