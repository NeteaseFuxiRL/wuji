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

import wuji.record
from . import Evaluator as _Evaluator

NAME = os.path.basename(os.path.splitext(__file__)[0])
module = importlib.import_module(f'wuji.rl.pth.{NAME}')


class Evaluator(_Evaluator):
    @staticmethod
    def ray_resources(config):
        return dict(num_cpus=0)

    def __init__(self, config, **kwargs):
        RL = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((module.RL,), map(wuji.parse.instance, filter(None, config.get('rl', 'wrap').split('\t') + config.get(NAME, 'wrap').split('\t')))))
        rl = RL(config, **{key: value for key, value in kwargs.items() if key != 'parallel'})
        super().__init__(config, rl, **kwargs)

    def get_hparam_real(self):
        def make(i, upper):
            return dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: self.rl.set_weight_loss(i, float(value)),
                get=lambda: np.array([self.rl.get_weight_loss(i)], np.float),
            )
        hparam = {}
        upper = self.config.getfloat('rl', 'discount_')
        if upper > 0:
            hparam[f'{NAME}/discount'] = dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: setattr(self.rl, 'discount', float(value)),
                get=lambda: np.array([self.rl.discount], np.float),
            )
        upper = self.config.getfloat('pg', 'prob_min_')
        if upper > 0:
            hparam[f'{NAME}/prob_min'] = dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: self.rl.set_prob_min(value),
                get=lambda: np.array([self.rl.get_prob_min()], np.float),
            )
        upper = self.config.getfloat('pg', 'prob_min_opponent_')
        if upper > 0:
            hparam[f'{NAME}/prob_min_opponent'] = dict(
                boundary=np.array([(0, upper)], np.float),
                set=lambda value: self.rl.set_prob_min_opponent(value),
                get=lambda: np.array([self.rl.get_prob_min_opponent()], np.float),
            )
        for i, key in enumerate(module.Losses._fields):
            upper = self.config.getfloat('ac_weight_loss', f'{key}_')
            if upper > 0:
                hparam['/'.join([NAME, f'weight_loss_{key}'])] = make(i, upper)
        return hparam

    def get_hparam_integer(self):
        hparam = {}
        upper = self.config.getint('ac', 'truncation_')
        if upper != 0:
            hparam[f'{NAME}/truncation'] = dict(
                boundary=np.array([(1, upper if upper > 0 else self.context['length'])], np.int),
                set=lambda value: self.rl.set_truncation(value),
                get=lambda: np.array([self.rl.get_truncation()], np.int),
            )
        return hparam

    def create_recorder(self):
        recorder = super().create_recorder()
        recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **{
            **{f'{NAME}/loss': self.outcome.loss},
            **{f'{NAME}/losses/{field}': getattr(self.outcome.losses, field) for field in self.outcome.losses._fields},
        }))
        return recorder
