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
import time
import itertools
import multiprocessing
import threading
import functools
import ast
import hashlib
import copy

import numpy as np
import matplotlib.pyplot as plt

from . import config, counter, parse, file, random, temp, ray


class Interval(object):
    def __init__(self):
        self.start = time.time()

    def close(self):
        self.interval = time.time() - self.start

    def get(self):
        return self.interval


def get_metric_db(config):
    root = os.path.expanduser(os.path.expandvars(config.get('model', 'root')))
    db = config.get('metric', 'db')
    return os.path.join(root, db)


def try_cast(s):
    try:
        return ast.literal_eval(s)
    except:
        return s


def abs_mean(data, dtype=np.float32):
    assert isinstance(data, np.ndarray), type(data)
    return np.sum(np.abs(data)) / dtype(data.size)


class RouletteWheel(object):
    def __init__(self, sizes):
        self.sizes = sizes - sizes.min() + np.finfo(sizes.dtype).eps
        self.rs = np.random.RandomState()

    def seed(self, seed):
        return self.rs.seed(seed)

    def __call__(self):
        total = self.sizes.sum()
        assert total > 0, self.sizes
        end = self.rs.uniform(0, total)
        seek = 0
        for i, size in enumerate(self.sizes):
            seek += size
            if seek >= end:
                return i
        raise RuntimeError(seek, end, total)


class Visualizer(multiprocessing.Process):
    def __init__(self, interval, wait=False):
        super().__init__()
        self.interval = interval
        self.wait = wait
        self.queue = multiprocessing.Queue()

    def close(self):
        self.queue.put(None)
        self.join()

    def draw(self, fig):
        with self.lock:
            data = copy.deepcopy(self.extract())
        if data is not None:
            args, kwargs = data
            self.plot(fig, *args, **kwargs)
            fig.show()

    def run(self):
        fig = self.create()
        self.lock = threading.Lock()
        receiver = threading.Thread(target=self.receive)
        receiver.start()
        while receiver.is_alive():
            self.draw(fig)
            plt.pause(self.interval)
        if self.wait:
            self.draw(fig)
            fig.canvas.set_window_title('stopped')
            plt.show()
        plt.close(fig)
        receiver.join()

    def receive(self):
        while True:
            try:
                args, kwargs = self.queue.get()
            except TypeError:
                break
            with self.lock:
                self.handle(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.queue.put(copy.deepcopy((args, kwargs)))

    def create(self):
        return plt.figure()

    def handle(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def extract(self):
        try:
            return self.args, self.kwargs
        except AttributeError:
            pass

    def plot(self, fig, *args, **kwargs):
        raise NotImplementedError()


def digest(decision, encoding):
    return hashlib.md5(bytes(itertools.chain(*[functools.reduce(lambda x, wrap: wrap(x), map(parse.instance, encoding['agent']['eval'])).serialize(decision[key]) if 'agent' in encoding else decision[key].tostring() for key, encoding in encoding.items()]))).hexdigest()


def nanmean(sample, *args, **kwargs):
    data0 = sample[0]
    if np.isscalar(data0) or type(data0) is np.ndarray:
        return np.nanmean(sample, *args, **kwargs)
    return type(data0)([nanmean([data[i] for data in sample], *args, **kwargs) for i, item in enumerate(data0)])
