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
import multiprocessing
import threading
import queue
import traceback
import signal
import pickle
import inspect

import humanfriendly
from tensorboardX import SummaryWriter

import wuji
import wuji.problem


class Recorder(object):
    def __init__(self, config, context, **kwargs):
        super().__init__()
        self.config_train = config
        self.config_test = wuji.config.test(pickle.loads(pickle.dumps(config)))
        self.context = context
        self.kwargs = kwargs
        self.task = []

    def register(self, timer, make, first=False):
        if isinstance(timer, str):
            timer = wuji.counter.Time(humanfriendly.parse_timespan(timer), first)
        elif isinstance(timer, int):
            timer = wuji.counter.Number(timer, first)
        self.task.append((timer, make))

    def __call__(self, force=False):
        for timer, make in self.task:
            if force or timer():
                try:
                    self.put(make())
                except:
                    traceback.print_exc()

    def put(self, record):
        raise NotImplementedError()


class Fake(Recorder):
    def close(self):
        pass

    def start(self):
        pass

    def put(self, record):
        pass


class Process(Recorder, multiprocessing.Process):
    def __init__(self, config, context, **kwargs):
        Recorder.__init__(self, config, context, **kwargs)
        multiprocessing.Process.__init__(self)
        self.queue = multiprocessing.Queue(config.getint('record', 'maxsize'))
        self._pid = os.getpid()

    def close(self):
        self.queue.put(None)

    def put(self, record):
        if callable(record):
            self.queue.put(record)

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)

    def run(self):
        try:
            self.create()
        except:
            traceback.print_exc()
            os.kill(self._pid, signal.SIGTERM)
            raise
        while self.tick() is not None:
            pass
        self.destroy()

    def tick(self):
        record = self.get()
        if record is None:
            return record
        try:
            record(self)
        except:
            traceback.print_exc()
        return record

    def create(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def destroy(self):
        pass

    def __repr__(self):
        return 'Process'


class Thread(Recorder, threading.Thread):
    def __init__(self, config, context, **kwargs):
        Recorder.__init__(self, config, context, **kwargs)
        threading.Thread.__init__(self)
        self.queue = queue.Queue(config.getint('record', 'maxsize'))
        self._pid = os.getpid()

    def close(self):
        self.queue.put(None)

    def put(self, record):
        if callable(record):
            self.queue.put(record)

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)

    def run(self):
        try:
            self.create()
        except:
            traceback.print_exc()
            os.kill(self._pid, signal.SIGTERM)
            raise
        while self.tick() is not None:
            pass
        self.destroy()

    def create(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def tick(self):
        record = self.get()
        if record is None:
            return record
        try:
            record(self)
        except:
            traceback.print_exc()
        return record

    def destroy(self):
        pass

    def __repr__(self):
        return 'Thread'


def tensorboard(recorder):
    __wrapper__ = inspect.stack()[0][3]

    class Recorder(recorder):
        def create(self):
            super().create()
            root_log = self.kwargs.get('root_log', os.path.join(self.kwargs['root'], 'log', self.kwargs['log']))
            os.makedirs(root_log, exist_ok=True)
            self.writer = SummaryWriter(root_log)
            cost = self.kwargs['cost']
            self.writer.add_text('config_test', wuji.config.text(self.config_test).replace('\n', '\n\n'), cost)
            self.writer.add_text('config_train', wuji.config.text(self.config_train).replace('\n', '\n\n'), cost)

        def destroy(self):
            self.writer.close()
            super().destroy()

        def __repr__(self):
            return f'{self.__module__}.{__wrapper__}<{super().__repr__()}>'
    return Recorder


def problem(recorder):
    __wrapper__ = inspect.stack()[0][3]

    class Recorder(recorder):
        def create(self):
            super().create()
            self.problem = wuji.problem.create(self.config_test)

        def destroy(self):
            self.problem.close()
            super().destroy()

        def __repr__(self):
            return f'{self.__module__}.{__wrapper__}<{super().__repr__()}>'
    return Recorder
