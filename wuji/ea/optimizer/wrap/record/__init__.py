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

import logging
import datetime
import functools
import inspect
import hashlib
import types
import time

import numpy as np

import wuji
import wuji.record


def create(optimizer):
    class Optimizer(optimizer):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            assert not hasattr(self, 'recorder')
            self.recorder = self.create_recorder()
            try:
                self.selection.record(self)
            except AttributeError:
                logging.warning('no record found in selection')
            self.recorder.start()

        def create_recorder(self):
            recorder = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.config.get('record', 'recorder').split('\t')))
            recorder = recorder(self.config, self.context, cost=self.cost, **self.kwargs)
            return recorder

        def close(self):
            logging.info('close')
            self.recorder.close()
            self.recorder.join()
            return super().close()

        def __call__(self, *args, **kwargs):
            try:
                return super().__call__(*args, **kwargs)
            finally:
                self.recorder()
    return Optimizer


def save(optimizer):
    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'save'), lambda: wuji.record.Save(self), self.config.getboolean('record', '_save'))
            return recorder
    return Optimizer


def profile(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def fetch(self):
        last = getattr(self, _name)
        t = time.time()
        elapsed = t - last.time + np.finfo(np.float).eps
        cost = (self.cost - last.cost) / elapsed
        setattr(self, _name, types.SimpleNamespace(time=t, cost=self.cost))
        return {
            'speed/total': cost,
            'speed/avg': cost / len(self),
        }

    class Optimizer(optimizer):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, types.SimpleNamespace(time=time.time(), cost=self.cost))

        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **fetch(self)))
            return recorder
    return Optimizer
