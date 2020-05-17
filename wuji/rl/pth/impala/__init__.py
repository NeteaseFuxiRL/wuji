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
import pickle
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
from .. import wrap as _wrap, pg, ac
from ..ac import Losses, Outcome

NAME = os.path.basename(os.path.dirname(__file__))


def cumulate(baseline, delta_vs, c, discount):
    vs = [baseline[-1] + delta_vs[-1]]
    for i in reversed(range(len(baseline) - 1)):
        vs.insert(0, baseline[i] + delta_vs[i] + discount * c[i] * (vs[0] - baseline[i + 1]))
    return torch.stack(vs)


class Actor(ac.RL):
    def seed(self, seed):
        return self.problem.seed(seed)

    def _norm(self, trajectory):
        reward = np.array([exp.reward for exp in trajectory])
        reward = self.norm(reward)
        return [exp._replace(reward=r) for exp, r in zip(trajectory, reward)]

    def _rollout(self):
        with torch.no_grad():
            trajectory, terminal, result = super().rollout()
        inputs = tuple(map(torch.cat, zip(*[exp['inputs'] for exp in trajectory])))
        prob = torch.cat([exp['prob'] for exp in trajectory])
        action = torch.cat([exp['action'] for exp in trajectory])
        reward = np.array([exp['reward'] for exp in trajectory])
        reward = self.norm(reward)
        reward = torch.FloatTensor(reward)
        return sum(exp.get('cost', 1) for exp in trajectory), (inputs, prob.gather(-1, action.view(-1, 1)).view(-1), action, reward, terminal), result


@wrap.call.any(*[name for name, func in inspect.getmembers(Actor, predicate=inspect.isroutine) if name.startswith('initialize_') or name.startswith('get_')])
@wrap.call.all(*[name for name, func in inspect.getmembers(Actor, predicate=inspect.isroutine) if name.startswith('set_')])
@wrap.call.all_async('set_blob', 'set_opponent_train')
@wrap.task
@_wrap.checkpoint
@_wrap.opponent.eval
@_wrap.opponent.train
@wrap.evaluate
@_wrap.optimizer
@_wrap.agent
@_wrap.model
@wrap.problem.context
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

    def make_tensors(self, inputs, _p, action, reward, terminal):
        logits, baseline = self.model(*inputs)
        prob, baseline = self.agent.prob(logits), baseline.view(-1)
        with torch.no_grad():
            p = prob.gather(-1, action.view(-1, 1)).view(-1)
            ratio = p / _p
            rho = ratio.clamp(max=self.rho)
            c = ratio.clamp(max=self.c)
            delta_vs = rho * (reward + torch.cat([self.discount * baseline[1:] - baseline[:-1], self.discount * terminal - baseline[-1:]]))
            vs = cumulate(baseline, delta_vs, c, self.discount)
            value = reward + self.discount * torch.cat([vs[1:], torch.FloatTensor([terminal])])
        return logits, prob, baseline, action, rho, value

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

    def get_losses(self, logits, prob, baseline, action, rho, value):
        advantage = value - baseline.detach()
        xe = self.agent.cross_entropy(logits=logits, prob=prob, action=action)
        entropy = (-torch.log(prob) * prob).sum(-1)
        return Losses(
            (rho * advantage * xe).mean(),
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
