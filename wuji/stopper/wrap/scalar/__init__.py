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

import inspect
import collections
import types
import hashlib

import numpy as np

import wuji.record


def ensure(stopper):
    class Stopper(stopper):
        def __call__(self, *args, **kwargs):
            scalar = kwargs['scalar']
            if isinstance(scalar, tuple):
                scalar = scalar[-1]
            try:
                scalar = scalar.item()
            except AttributeError:
                pass
            kwargs['scalar'] = scalar
            return super().__call__(*args, **kwargs)
    return Stopper


def smooth(sample='sample/train', begin=None, stride=None, method=np.mean, label='smooth'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, _name)
                maxlen = self.config.getint(*sample.split('/')) if isinstance(sample, str) else sample
                setattr(self, _name, types.SimpleNamespace(
                    recent=collections.deque(maxlen=maxlen),
                    begin=maxlen if begin is None else self.config.getint(*begin.split('/')) if isinstance(begin, str) else begin,
                    stride=wuji.counter.Number(maxlen if stride is None else self.config.getint(*stride.split('/')) if isinstance(stride, str) else stride),
                    record=self.config.getboolean('stopper', 'record')
                ))

            def __call__(self, *args, **kwargs):
                scalar = kwargs['scalar']
                attr = getattr(self, _name)
                attr.recent.append(scalar)
                if len(attr.recent) >= attr.begin and attr.stride():
                    scalar = method(attr.recent)
                    kwargs['scalar'] = scalar
                    if attr.record:
                        self.evaluator.recorder.put(wuji.record.Scalar(self.evaluator.cost, **{label: scalar}))
                    return super().__call__(*args, **kwargs)
                return False
        return Stopper
    return decorate


def greater(value='stopper/scalar'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, _name)
                setattr(self, _name, eval(self.config.get(*value.split('/'))) if isinstance(value, str) else value)

            def __call__(self, *args, **kwargs):
                scalar = kwargs['scalar']
                if scalar > getattr(self, _name):
                    return True
                return super().__call__(*args, **kwargs)
        return Stopper
    return decorate
