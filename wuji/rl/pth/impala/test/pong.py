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
import operator

import numpy as np
import torch
import ray

import wuji.problem
from .. import RL, wrap

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
    ray.init(**wuji.ray.init(config))
    wuji.random.seed(0)
    rl = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((RL,), map(wuji.parse.instance, filter(None, config.get('rl', 'wrap').split('\t') + config.get(NAME, 'wrap').split('\t')))))
    rl = wrap.hook.make_tensors(rl)
    with contextlib.closing(rl(config)) as rl:
        ray.get([actor.seed.remote(0) for actor in rl.actor])
        check_text('model', '\n'.join(['\t'.join([key, hashlib.md5(value.numpy().tostring()).hexdigest()]) for key, value in rl.get_blob().items()]))
        outcome = rl()
        inputs, _p, action, reward, terminal = zip(*[attr.inputs for attr in rl.hook_make_tensors])
        state = torch.cat(list(map(operator.itemgetter(0), inputs)))
        _p, reward = map(torch.cat, (_p, reward))
        terminal = torch.stack(terminal)
        logits, prob, baseline, action, rho, value = map(torch.cat, zip(*[attr.outputs for attr in rl.hook_make_tensors]))
        # trajectory
        check_text('state', '\n'.join([hashlib.md5(s.numpy().tostring()).hexdigest() for s in state]))
        check_text('logits', array2tsv(logits.detach().numpy()))
        check_text('prob', array2tsv(prob.detach().numpy()))
        check_text('baseline', array2tsv(baseline.detach().numpy()))
        check_text('action', array2tsv(action.detach().numpy()))
        check_text('reward', array2tsv(reward.detach().numpy()))
        check_text('terminal', array2tsv(terminal.detach().numpy()))
        check_text('_p', array2tsv(_p.detach().numpy()))
        check_text('rho', array2tsv(rho.detach().numpy()))
        check_text('value', array2tsv(value.detach().numpy()))
        # loss
        for key, value in outcome.losses._asdict().items():
            check_text(os.path.join('loss', key), str(value.item()))
        check_text('loss', str(outcome.loss.item()))
