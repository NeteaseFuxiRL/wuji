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
import functools
import itertools
import configparser
import logging

import numpy as np
import torch.cuda
import ray
import psutil

import wuji
from . import wrap
from .. import wrap as _wrap, pg, ac, impala
from ..ac import Losses, Outcome

NAME = os.path.basename(os.path.dirname(__file__))


class Actor(ac.RL):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.norm_advantage = eval('lambda advantage: ' + self.config.get(NAME, 'norm_advantage'))

    def seed(self, seed):
        return self.problem.seed(seed)

    def _rollout(self):
        with torch.no_grad():
            trajectory, terminal, result = self.rollout()
        inputs = tuple(map(torch.cat, zip(*[exp['inputs'] for exp in trajectory])))
        prob = torch.cat([exp['prob'] for exp in trajectory])
        baseline = torch.cat([exp['baseline'] for exp in trajectory]).view(-1)
        action = torch.cat([exp['action'] for exp in trajectory])
        reward = np.array([exp['reward'] for exp in trajectory])
        reward = self.norm(reward)
        reward = torch.FloatTensor(reward)
        value = pg.cumulate(reward, self.discount, terminal)
        advantage = self.norm_advantage(value - baseline)
        p = prob.gather(-1, action.view(-1, 1)).view(-1)
        return sum(exp.get('cost', 1) for exp in trajectory), (inputs, action, value, advantage, p), result


@impala.wrap.call.any(*[name for name, func in inspect.getmembers(Actor, predicate=inspect.isroutine) if name.startswith('initialize_') or name.startswith('get_')])
@impala.wrap.call.all(*[name for name, func in inspect.getmembers(Actor, predicate=inspect.isroutine) if name.startswith('set_')])
@impala.wrap.call.all_async('set_blob', 'set_opponent_train')
@impala.wrap.task
@_wrap.checkpoint
@_wrap.opponent.eval
@_wrap.opponent.train
@impala.wrap.evaluate
@_wrap.optimizer
@_wrap.agent
@_wrap.model
@impala.wrap.problem.context
@wrap.attr
@ac.wrap.attr
@pg.wrap.problem.batch_size
@pg.wrap.agent
@pg.wrap.attr
class RL(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.actor = self.create_actor()
        self.task = []
        self.async_call = [{} for _ in range(len(self.actor))]

    def close(self):
        return ray.get([actor.close.remote() for actor in self.actor])

    def create_actor(self):
        try:
            parallel = self.config.getint(NAME, 'parallel')
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            parallel = psutil.cpu_count(logical=False)
            logging.warning(f'number of actors not configured, use default {parallel}')
        actor = functools.reduce(lambda x, wrap: wrap(x), itertools.chain((self.kwargs.pop('actor', Actor),), map(wuji.parse.instance, filter(None, self.config.get('rl', 'wrap').split('\t')))))
        actor = _wrap.remote_actor(actor, num_cpus=1)
        config = pickle.dumps(self.config)
        return [actor.remote(config, {'index': index, **self.kwargs}) for index in range(parallel)]

    def __len__(self):
        return len(self.actor)

    def __next__(self):
        (ready,), _ = ray.wait(self.task)
        cost, tensors, result = ray.get(ready)
        index = self.task.index(ready)
        actor = self.actor[index]
        for key, (args, kwargs) in self.async_call[index].items():
            getattr(actor, key).remote(*args, **kwargs)
        self.async_call[index] = {}
        self.task[index] = actor._rollout.remote()
        return cost, self.make_tensors(*tensors), result

    def make_tensors(self, inputs, action, value, advantage, _p):
        logits, baseline = self.model(*inputs)
        prob, baseline = self.agent.prob(logits), baseline.view(-1)
        with torch.no_grad():
            p = prob.gather(-1, action.view(-1, 1)).view(-1)
            ratio = p / _p.clamp(min=np.finfo(np.float32).eps)
        return prob, baseline, value, advantage, ratio

    def get_losses(self, prob, baseline, value, advantage, ratio):
        policy = -torch.min(
            advantage * ratio,
            advantage * ratio.clamp(min=1 - self.epsilon, max=1 + self.epsilon),
        )
        entropy = (-torch.log(prob) * prob).sum(-1)
        return Losses(
            policy.mean(),
            self.loss_critic(baseline, value),
            entropy.mean(),
        )

    def total_loss(self, policy, critic, entropy):
        return (torch.stack([policy, critic, -entropy]) * self.weight_loss).sum()

    def backward(self):
        cost, tensors, result = self.__next__()
        losses = self.get_losses(*tensors)
        loss = self.total_loss(*losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad_norm()
        return Outcome(cost, loss, losses, result)

    def __call__(self):
        outcome = self.backward()
        self.optimizer.step()
        _blob = ray.put(self.get_blob())
        for async_call in self.async_call:
            async_call['set_blob'] = ((_blob,), {})
        return outcome
