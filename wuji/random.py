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

import random
import pickle
import hashlib
import logging

import numpy as np
import torch


def print_state(prefix='', print=logging.warning):
    print(prefix + 'random state: ' + hashlib.md5(pickle.dumps(random.getstate())).hexdigest())
    print(prefix + 'numpy random state: ' + hashlib.md5(pickle.dumps(np.random.get_state())).hexdigest())
    print(prefix + 'torch random state: ' + hashlib.md5(torch.random.get_rng_state().numpy().tostring()).hexdigest())


def seed(seed, prefix='', print=logging.warning):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if prefix:
        print_state(prefix=prefix, print=print)
    return seed
