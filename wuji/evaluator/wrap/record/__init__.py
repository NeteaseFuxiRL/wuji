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

import datetime
import logging
import inspect
import hashlib
import types
import time
import collections

import numpy as np

import wuji.record
from . import outcome


def profile(evaluator):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def fetch(self):
        last = getattr(self, _name)
        t = time.time()
        speed = (self.cost - last.cost) / (t - last.time + np.finfo(np.float).eps)
        setattr(self, _name, types.SimpleNamespace(cost=self.cost, time=t))
        return {
            'speed/total': speed,
        }

    class Evaluator(evaluator):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, types.SimpleNamespace(cost=self.cost, time=time.time()))

        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **fetch(self)))
            return recorder
    return Evaluator


def save(evaluator):
    class Evaluator(evaluator):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'save'), lambda: wuji.record.Save(self), self.config.getboolean('record', '_save'))
            return recorder
    return Evaluator


def metric(evaluator):
    class Evaluator(evaluator):
        def create_recorder(self):
            recorder = super().create_recorder()
            metric = []
            try:
                metric += self.context['metric']
            except KeyError:
                logging.warning('metric not found in context')
            try:
                metric += self.metric
            except AttributeError:
                pass
            if metric:
                recorder.register(self.config.get('record', 'metric'), lambda: wuji.record.Metric(metric, **{
                    'name': self.kwargs['name'],
                    'blob': self.get_blob(),
                    'now': datetime.datetime.now(),
                    'duration': self.duration,
                    'iteration': self.iteration,
                    'cost': self.cost,
                }))
            return recorder
    return Evaluator


def blob(evaluator):
    class Evaluator(evaluator):
        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'histogram_dict'), lambda: wuji.record.HistogramDict(self.cost, **collections.OrderedDict([('blob/' + key, var.cpu().clone()) for key, var in self.get_blob().items()])))
            return recorder
    return Evaluator
