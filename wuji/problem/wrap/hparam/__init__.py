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
import itertools
import collections

import numpy as np


def decision(problem):
    name = os.path.basename(os.path.dirname(__file__))

    class Problem(problem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, name)
            setattr(self, name, getattr(self, 'get_' + name)())
            if 'encoding' not in self.context:
                self.context['encoding'] = {}
            for coding, hparam in getattr(self, name).items():
                if hparam:
                    assert coding not in self.context['encoding']
                    self.context['encoding'][coding] = dict(
                        boundary=np.concatenate([hparam['boundary'] for hparam in hparam.values()]),
                        header=list(itertools.chain(*[[f'{key}{i}' for i in range(len(hparam['boundary']))] if len(hparam['boundary']) > 1 else [key] for key, hparam in hparam.items()])),
                    )

        def get_hparam(self):
            return {method[11:]: getattr(self, method)() for method in dir(self) if method.startswith('get_hparam_')}

        def initialize_real(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            coding = method.rsplit('_', 1)[-1]
            boundary = [hparam['boundary'] for hparam in getattr(self, name).get(coding, {}).values()]
            if boundary:
                lower, upper = np.concatenate(boundary).T
                return lower + np.random.random(lower.shape) * (upper - lower)
            else:
                return np.array([], np.float)

        def set_real(self, value):
            assert isinstance(value, collections.Iterable), type(value)
            method = inspect.getframeinfo(inspect.currentframe()).function
            coding = method.rsplit('_', 1)[-1]
            for key, hparam in getattr(self, name).get(coding, {}).items():
                hparam['set'](*[next(value) for _ in hparam['boundary']])
            return value

        def get_real(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            coding = method.rsplit('_', 1)[-1]
            decision = [hparam['get']() for hparam in getattr(self, name).get(coding, {}).values()]
            if decision:
                return np.concatenate(decision)
            else:
                return np.array([], np.float)

        def initialize_integer(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            coding = method.rsplit('_', 1)[-1]
            boundary = [hparam['boundary'] for hparam in getattr(self, name).get(coding, {}).values()]
            if boundary:
                lower, upper = np.concatenate(boundary).T
                return np.array([random.randint(l, u) for l, u in zip(lower, upper)])
            else:
                return np.array([], np.int)

        def set_integer(self, value):
            assert isinstance(value, collections.Iterable), type(value)
            method = inspect.getframeinfo(inspect.currentframe()).function
            coding = method.rsplit('_', 1)[-1]
            for key, hparam in getattr(self, name).get(coding, {}).items():
                hparam['set'](*[next(value) for _ in hparam['boundary']])
            return value

        def get_integer(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            coding = method.rsplit('_', 1)[-1]
            decision = [hparam['get']() for hparam in getattr(self, name).get(coding, {}).values()]
            if decision:
                return np.concatenate(decision)
            else:
                return np.array([], np.int)
    return Problem
