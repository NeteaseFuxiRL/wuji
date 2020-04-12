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
import hashlib
import types
import time

import numpy as np
import torch
import humanfriendly

import wuji.record


def record(label='stopper'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name + label).encode()).hexdigest()

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, _name)
                setattr(self, _name, wuji.counter.Time(humanfriendly.parse_timespan(self.config.get('record', 'scalar'))))

            def __call__(self, *args, **kwargs):
                scalar = kwargs['scalar']
                if getattr(self, _name)():
                    self.evaluator.recorder.put(wuji.record.Scalar(self.evaluator.cost, **{label: scalar}))
                return super().__call__(*args, **kwargs)
        return Stopper
    return decorate


def csv(relpath='stopper', keep=0, model=False, interval='1h'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name + relpath).encode()).hexdigest()

    def decorate(stopper):
        class Stopper(stopper):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, _name)
                root = os.path.expanduser(os.path.expandvars(os.path.join(self.evaluator.kwargs['root_log'], relpath)))
                os.makedirs(root, exist_ok=True)
                attr = types.SimpleNamespace(
                    table=[('Wall time', 'Step', 'Value')],
                    prefix=os.path.join(root, str(self.evaluator.cost)),
                    interval=wuji.counter.Time(humanfriendly.parse_timespan(interval)) if isinstance(interval, str) else lambda: False,
                )
                if model:
                    torch.save(self.evaluator.__getstate__(), attr.prefix + '.pth')
                setattr(self, _name, attr)

            def close(self):
                attr = getattr(self, _name)
                np.savetxt(attr.prefix + '.csv', attr.table, fmt='%s', delimiter=',')
                if keep > 0:
                    wuji.file.tidy(os.path.dirname(attr.prefix), keep)
                return super().close()

            def __call__(self, *args, **kwargs):
                scalar = kwargs['scalar']
                attr = getattr(self, _name)
                attr.table.append((time.time(), self.evaluator.cost, scalar))
                if attr.interval():
                    np.savetxt(attr.prefix + '.csv', attr.table, fmt='%s', delimiter=',')
                return super().__call__(*args, **kwargs)
        return Stopper
    return decorate
