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
import hashlib
import copy
import types

import numpy as np

import wuji.record


def best(label='checkpoint'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, _name)
                setattr(self, _name, types.SimpleNamespace(
                    best=np.finfo(np.float).min,
                ))

            def __call__(self, *args, **kwargs):
                scalar = kwargs['scalar']
                attr = getattr(self, _name)
                if scalar > attr.best:
                    attr.best = scalar
                    attr.decision = copy.deepcopy(self.evaluator.get())
                    if label:
                        self.evaluator.recorder.put(wuji.record.Scalar(self.evaluator.cost, **{label: scalar}))
                terminate = super().__call__(*args, **kwargs)
                if terminate:
                    self.evaluator.set(attr.decision)
                return terminate
        return Stopper
    return decorate
