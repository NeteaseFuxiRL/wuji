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
import importlib
import itertools
import functools

import numpy as np
import torch.cuda

import wuji.record
from . import Evaluator as _Evaluator

NAME = os.path.basename(os.path.splitext(__file__)[0])
module = importlib.import_module(f'wuji.rl.pth.{NAME}')


class Evaluator(_Evaluator):
    @staticmethod
    def ray_resources(config):
        return dict(num_cpus=1)

    def __init__(self, config, **kwargs):
        RL = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((module.RL,), map(wuji.parse.instance, filter(None, config.get('rl', 'wrap').split('\t') + config.get(NAME, 'wrap').split('\t')))))
        rl = RL(config, **{key: value for key, value in kwargs.items() if key != 'parallel'})
        super().__init__(config, rl, **kwargs)

    def get_hparam_real(self):
        hparam = {}
        upper = self.config.getfloat('train', 'lr_')
        if upper > 0:
            hparam[f'{NAME}/lr'] = dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: self.rl.set_lr(value),
                get=lambda: np.array([self.rl.get_lr()[0]], np.float),
            )
        upper = self.config.getfloat('rl', 'discount_')
        if upper > 0:
            hparam[f'{NAME}/discount'] = dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: setattr(self.rl, 'discount', float(value)),
                get=lambda: np.array([self.rl.discount], np.float),
            )
        upper = self.config.getfloat(NAME, 'epsilon_')
        if upper > 0:
            hparam[f'{NAME}/epsilon'] = dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: setattr(self.rl.agent, 'epsilon', float(value)),
                get=lambda: np.array([self.rl.agent.epsilon], np.float),
            )
        return hparam

    def get_hparam_integer(self):
        hparam = {}
        upper = self.config.getfloat('train', 'batch_size_')
        if upper > 1:
            hparam['batch_size'] = dict(
                boundary=np.array([(1, upper)], np.int),
                set=lambda value: setattr(self.rl, 'batch_size', value),
                get=lambda: np.array([self.rl.batch_size], np.int),
            )
        return hparam

    def create_recorder(self):
        recorder = super().create_recorder()
        recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **{
            **{f'{NAME}/loss': self.outcome.loss.item()},
        }))
        return recorder
