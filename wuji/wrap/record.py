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
import types
import time

import numpy as np
import psutil

import wuji.record


def memory(optimizer):
    class Optimizer(optimizer):
        def create_recorder(self):
            def uss(proc):
                try:
                    return proc.memory_full_info().uss
                except:
                    return 0
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'memory'), lambda: wuji.record.Scalar(self.cost, **{
                'memory/uss': sum(uss(proc) for proc in psutil.process_iter()),
            }))
            return recorder
    return Optimizer


def profile(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class Optimizer(optimizer):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, types.SimpleNamespace(time=time.time(), cost=self.cost))

        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **self.speed()))
            return recorder

        def speed(self):
            last = getattr(self, _name)
            t = time.time()
            elapsed = t - last.time + np.finfo(np.float).eps
            cost = (self.cost - last.cost) / elapsed
            setattr(self, _name, types.SimpleNamespace(time=t, cost=self.cost))
            return {
                'speed/cost/total': cost,
                'speed/cost/avg': cost / len(self),
            }
    return Optimizer
