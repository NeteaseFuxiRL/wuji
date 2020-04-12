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

import numpy as np

from .. import Stopper as _Stopper


class Patience(_Stopper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = self.config.getint('stopper', 'patience')
        self.count = 0
        self.last = np.finfo(np.float).min

    def __call__(self, *args, **kwargs):
        scalar = kwargs['scalar']
        if scalar <= self.last:
            self.count += 1
            if self.count >= self.patience:
                return True
        else:
            self.count = 0
        self.last = scalar
        return False


def greater(value='stopper/scalar'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class Stopper(_Stopper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, eval(self.config.get(*value.split('/'))) if isinstance(value, str) else value)

        def __call__(self, *args, **kwargs):
            scalar = kwargs['scalar']
            if scalar > getattr(self, _name):
                return True
            return False
    return Stopper
