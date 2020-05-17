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
import pickle
import inspect
import itertools
import asyncio
import functools
import contextlib
import configparser
import warnings

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
import ray

import wuji.problem.mdp
from . import opponent


def remote_actor(actor, *args, **kwargs):
    wrap = ray.remote(*args, **kwargs) if args or kwargs else ray.remote
    @wrap
    class Actor(actor):
        def __init__(self, config, kwargs):
            torch.set_num_threads(1)
            config = pickle.loads(config)
            try:
                index = kwargs['index']
                seed = config.getint('config', 'seed') + index
                wuji.random.seed(seed, prefix=f'actor[{index}] seed={seed}: ')
            except configparser.NoOptionError:
                pass
            if 'num_gpus' in kwargs and kwargs['num_gpus'] > 0:
                try:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, ray.get_gpu_ids()))
                    type(self).__name__ = 'GPU' + os.environ['CUDA_VISIBLE_DEVICES']
                except:
                    type(self).__name__ = 'GPU'
                device = torch.device('cuda')
            else:
                type(self).__name__ = 'CPU'
                device = torch.device('cpu')
            kwargs.pop('device', None)
            super().__init__(config, device=device, **kwargs)
    return Actor


def checkpoint(rl):
    class RL(rl):
        def __getstate__(self):
            return dict(context=self.get_context(), decision=dict(blob=self.get_blob()))

        def __setstate__(self, state):
            decision = state.pop('decision')
            self.set_blob(decision['blob'])
            state.pop('context', None)
    return RL


def problem(rl):
    class RL(rl):
        def __init__(self, config, *args, **kwargs):
            assert not hasattr(self, 'problem')
            self.config = config
            self.problem = wuji.problem.seed(wuji.problem.create(config), **kwargs)
            self.context = self.problem.context
            self.transform = lambda x: to_tensor(x) if len(x.shape) > 2 else torch.FloatTensor(x)
            super().__init__(config, *args, **kwargs)

        def close(self):
            try:
                return super().close()
            finally:
                self.problem.close()

        def get_context(self):
            return self.context

        def initialize_real(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method, lambda: np.array([], np.float))()

        def set_real(self, value):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method, lambda value: value)(value)

        def get_real(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method, lambda: np.array([], np.float))()

        def initialize_integer(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method, lambda: np.array([], np.int))()

        def set_integer(self, value):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method, lambda value: value)(value)

        def get_integer(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method, lambda: np.array([], np.int))()

        def set_name_reward(self, value):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)(value)

        def get_name_reward(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)()

        def set_weight_reward(self, value):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)(value)

        def get_weight_reward(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)()

        def set_name_final_reward(self, value):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)(value)

        def get_name_final_reward(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)()

        def set_weight_final_reward(self, value):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)(value)

        def get_weight_final_reward(self):
            method = inspect.getframeinfo(inspect.currentframe()).function
            return getattr(self.problem, method)()
    return RL


def model(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, 'model')
            self.set_kind(kwargs.get('kind', 0))

        def set_kind(self, kind):
            self.kind = kind
            module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in self.context['encoding']['blob']['module']))
            self.model = module(self.config, **self.context['encoding']['blob']['init'][self.kind]['kwargs'])
            self.model.train()
            self.header = list(self.model.state_dict().keys())

        def get_kind(self):
            return self.kind

        def initialize_blob(self):
            module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in self.context['encoding']['blob']['module']))
            return module(self.config, **self.context['encoding']['blob']['init'][self.kind]['kwargs']).state_dict()

        def set_blob(self, blob):
            return self.model.set_blob(blob)

        def get_blob(self):
            return self.model.get_blob()
    return RL


def agent(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, 'agent')
            self.agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.context['encoding']['blob']['agent']['train']))(self.model)

        def get_agent(self):
            return self.agent

        def __call__(self, *args, **kwargs):
            assert self.model is self.agent.model
            return super().__call__(*args, **kwargs)
    return RL


def optimizer(rl):
    def create(config, model):
        return eval('lambda params, lr: ' + config.get('train', 'optimizer'))(filter(lambda p: p.requires_grad, model.parameters()), config.getfloat('train', 'lr'))

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, 'optimizer')
            try:
                clip_grad_norm = self.config.getfloat('train', 'clip_grad_norm')
                self.clip_grad_norm = lambda: nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
            except configparser.NoOptionError:
                self.clip_grad_norm = lambda: None
            self.optimizer = create(self.config, self.model)

        def set_lr(self, lr):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        def get_lr(self):
            return [group['lr'] for group in self.optimizer.param_groups]

        def __call__(self, *args, **kwargs):
            assert next(self.model.parameters()) is self.optimizer.param_groups[0]['params'][0]
            return super().__call__(*args, **kwargs)
    return RL


def evaluate(rl):
    from wuji.rl.pth.wrap.opponent import make_agents

    class RL(rl):
        def evaluate(self):
            with torch.no_grad():
                try:
                    self.model.eval()
                    agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.context['encoding']['blob']['agent']['eval']))(self.model)
                    cost = 0
                    results = []
                    sample = self.config.getint('sample', 'eval')
                    opponents = self.get_opponents_eval()
                    if sample < len(opponents):
                        warnings.warn(f'sample={sample} is lesser than the number of opponents ({len(opponents)})')
                    for seed, opponent in zip(range(sample), itertools.cycle(opponents)):
                        opponent = make_agents(self.config, self.context['encoding']['blob'], opponent)
                        with contextlib.closing(self.problem.evaluating(seed)):
                            controllers, ticks = self.problem.reset(self.kind, *opponent)
                            costs = asyncio.get_event_loop().run_until_complete(asyncio.gather(
                                wuji.problem.mdp._rollout(controllers[0], agent),
                                *[wuji.problem.mdp._rollout(controller, agent) for controller, agent in zip(controllers[1:], opponent.values())],
                                *map(wuji.problem.mdp.ticking, ticks),
                            ))[:len(controllers)]
                        cost += max(costs)
                        results.append(controllers[0].get_result())
                    return cost, self.evaluate_reduce(results)
                finally:
                    self.model.train()

        def evaluate_map(self, seed, opponent):
            with torch.no_grad(), contextlib.closing(self.problem.evaluating(seed)):
                try:
                    self.model.eval()
                    agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.context['encoding']['blob']['agent']['eval']))(self.model)
                    opponent = make_agents(self.config, self.context['encoding']['blob'], opponent)
                    controllers, ticks = self.problem.reset(self.kind, *opponent)
                    costs = asyncio.get_event_loop().run_until_complete(asyncio.gather(
                        wuji.problem.mdp._rollout(controllers[0], agent),
                        *[wuji.problem.mdp._rollout(controller, agent) for controller, agent in zip(controllers[1:], opponent.values())],
                        *map(wuji.problem.mdp.ticking, ticks),
                    ))[:len(controllers)]
                    result = controllers[0].get_result()
                    return seed, max(costs), result
                finally:
                    self.model.train()

        def evaluate_reduce(self, results):
            return self.problem.evaluate_reduce(results)
    return RL
