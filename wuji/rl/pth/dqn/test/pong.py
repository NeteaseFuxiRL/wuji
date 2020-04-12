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
import io
import functools
import itertools
import configparser
import contextlib
import hashlib

import numpy as np
import torch

import wuji.problem
from .. import RL
from ... import pg

PREFIX = os.path.splitext(__file__)[0]
NAME = __file__.split(os.sep)[-3]


def array2tsv(a):
    s = io.BytesIO()
    np.savetxt(s, a, fmt='%s', delimiter='\t')
    return s.getvalue().decode()


def check_text(name, value, ext='.tsv'):
    assert isinstance(value, str), type(value)
    path = os.path.join(PREFIX, f'{name}{ext}')
    try:
        with open(path, 'r') as f:
            assert f.read() == value
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(value)


def test():
    config = configparser.ConfigParser()
    config.read(PREFIX + '.ini')
    wuji.random.seed(0)
    rl = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((RL,), map(wuji.parse.instance, filter(None, config.get('rl', 'wrap').split('\t') + config.get('dqn', 'wrap').split('\t')))))
    rl = pg.wrap.hook.rollout(rl)
    with contextlib.closing(rl(config)) as rl:
        rl.problem.seed(0)
        check_text('model', '\n'.join(['\t'.join([key, hashlib.md5(value.numpy().tostring()).hexdigest()]) for key, value in rl.get_blob().items()]))
        outcome = rl()
        (trajectory,), (result,) = zip(*rl.hook_rollout)
        # trajectory
        check_text(os.path.join('trajectory', 'state'), '\n'.join([hashlib.md5(exp['inputs'][0].detach().numpy().tostring()).hexdigest() for exp in trajectory]))
        check_text(os.path.join('trajectory', 'action'), '\n'.join([str(exp['action'].item()) for exp in trajectory]))
        check_text(os.path.join('trajectory', 'reward'), '\n'.join([str(exp['reward'].item()) for exp in trajectory]))
        # tensors
        value = rl.value(trajectory)
        check_text(os.path.join('tensors', 'value'), array2tsv(value.numpy()))
        # loss
        check_text('loss', str(outcome.loss.item()))
