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
import types
import functools
import itertools
import multiprocessing
import queue
import configparser
import logging
import traceback

import numpy as np
import torch
import psutil

import wuji
from . import wrap
from .. import Actor, NAME
from ... import wrap as _wrap
from ...ac import Outcome


class CycleQueue(object):
    def __init__(self, *args):
        self.queues = itertools.cycle(args)
        self.queue = next(self.queues)

    def get(self, *args, **kwargs):
        while True:
            try:
                return self.queue.get(*args, **kwargs)
            except queue.Empty:
                self.queue = next(self.queues)


class _Actor(multiprocessing.Process):
    def __init__(self, config, comm, **kwargs):
        super().__init__()
        self.config = config
        self.comm = comm
        self.kwargs = kwargs
        self.cmd = multiprocessing.Queue()
        self.ready = multiprocessing.Event()
        self.queue = CycleQueue(self.comm.cmd, self.cmd)
        self.timeout = 0.01

    def run(self):
        torch.set_num_threads(1)
        actor = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((self.kwargs.pop('actor', Actor),), map(wuji.parse.instance, filter(None, self.config.get('rl', 'wrap').split('\t') + self.config.get(NAME, 'wrap').split('\t')))))
        actor = actor(self.config, **self.kwargs)
        self.ready.set()
        index_actor = self.kwargs['index_actor']
        while True:
            name, args, kwargs = self.queue.get(timeout=self.timeout)
            # print(self.kwargs['index'], name)
            func = getattr(actor, name)
            try:
                result = func(*args, **kwargs)
                self.comm.result[name].put((index_actor, result))
            except Exception as e:
                traceback.print_exc()
                self.comm.result[name].put((index_actor, e))


@_wrap.optimizer
@_wrap.model
class _RL(object):
    @staticmethod
    def get_parallel(config, **kwargs):
        try:
            parallel = kwargs.pop('parallel', config.getint(NAME, 'parallel'))
            logging.info(f'{NAME}: parallel={parallel}')
        except (configparser.NoOptionError, ValueError):
            parallel = psutil.cpu_count(logical=False)
            logging.warning(f'{NAME}: parallel={parallel} (default)')
        return parallel

    @staticmethod
    def ray_resources(config, **kwargs):
        return dict(
            num_cpus=_RL.get_parallel(config, **kwargs),
        )

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        parallel = _RL.get_parallel(config, **kwargs)
        actor = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((self.kwargs.pop('actor', Actor),), map(wuji.parse.instance, filter(None, self.config.get('rl', 'wrap').split('\t')))))
        self.comm = types.SimpleNamespace(
            cmd=multiprocessing.Queue(),
            actor=[types.SimpleNamespace(ready=multiprocessing.Event(), cmd=multiprocessing.Queue()) for _ in range(parallel)],
            result={name: multiprocessing.Queue() for name, func in inspect.getmembers(actor, predicate=inspect.isroutine)},
        )
        self.actor = self.create_actor(parallel, **kwargs)
        logging.info(f'{NAME.upper()}[{self.kwargs.get("index", 0)}]: {tuple(actor.kwargs["index"] for actor in self.actor)}')
        for actor in self.actor:
            actor.start()
        for actor in self.actor:
            actor.ready.wait()
        self.async_put('get_context')
        _, self.context = self.async_get('get_context')
        torch.set_num_threads(1)

    def close(self):
        for actor in self.actor:
            actor.terminate()

    def create_actor(self, parallel, **kwargs):
        begin = kwargs.pop('index', 0) * parallel
        try:
            seed = kwargs.pop('seed')
        except KeyError:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        def make(index_actor):
            index = begin + index_actor
            return _Actor(self.config, self.comm, index_actor=index_actor, index=index, seed=seed + index, **kwargs)
        return [make(index_actor) for index_actor in range(parallel)]

    def __len__(self):
        return len(self.actor)

    def async_put(self, name, *args, **kwargs):
        return self.comm.cmd.put((name, args, kwargs))

    def async_get(self, name):
        index_actor, result = self.comm.result[name].get()
        if isinstance(result, Exception):
            raise result
        return index_actor, result

    def broadcast_put(self, name, *args, **kwargs):
        for actor in self.actor:
            actor.cmd.put((name, args, kwargs))

    def broadcast_get(self, name):
        _, results = zip(*[self.comm.result[name].get() for _ in self.actor])
        exceptions = [result for result in results if isinstance(result, Exception)]
        if exceptions:
            raise exceptions[0]
        return results[0]

    def broadcast(self, name, *args, **kwargs):
        self.broadcast_put(name, *args, **kwargs)
        return self.broadcast_get(name)


@wrap.call.any(*[name for name, func in inspect.getmembers(Actor, predicate=inspect.isroutine) if name.startswith('initialize_') or name.startswith('get_')])
@wrap.call.all(*[name for name, func in inspect.getmembers(Actor, predicate=inspect.isroutine) if name.startswith('set_') and not hasattr(_RL, name)])
@wrap.async_reset('set_blob')
@wrap.async_count
@wrap.evaluate
@wrap.opponent
@_wrap.checkpoint
@_wrap.opponent.eval
@_wrap.opponent.train
class RL(_RL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_kwargs = [{} for _ in range(len(self))]

    def __call__(self):
        idle = len(self) - self.async_count.get('gradient', 0)
        if idle > 0:
            blob = self.model.get_blob()
            opponent = self.get_opponent_train()
            for _ in range(idle):
                self.async_put('gradient', blob, set_opponent_train=((opponent,), {}))
        index_actor, outcome = self.async_get('gradient')
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), outcome.gradient):
            param.grad = grad
        self.optimizer.step()
        self.actor[index_actor].cmd.put(('gradient', (self.model.get_blob(),), self.gradient_kwargs[index_actor]))
        self.async_count['gradient'] += 1
        self.gradient_kwargs[index_actor] = {}
        return Outcome(outcome.cost, outcome.loss, outcome.losses, outcome.result)
