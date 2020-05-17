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
import asyncio
import collections

import numpy as np
import torch
import recordtype

import wuji.problem.mdp
import wuji.model.pth
from .. import pg, wrap as _wrap
from . import wrap, agent

NAME = os.path.basename(os.path.dirname(__file__))
Losses = collections.namedtuple('Losses', ['policy', 'critic', 'entropy'])
Outcome = recordtype.recordtype('Outcome', ['cost', 'loss', 'losses', 'result'])


def cumulate(reward, discount, terminal=0):
    value = []
    v = terminal
    for r, d in zip(reversed(torch.unbind(reward)), reversed(torch.unbind(discount))):
        v = v * d + r
        value.insert(0, v)
    value = torch.stack(value)
    assert value.size() == reward.size(), value.size()
    return value


def gae(reward, discount, baseline, lmd=0.95):
    value = []
    v = 0
    for i in reversed(range(len(reward))):
        delta = reward[i] + discount[i] * baseline[i + 1] - baseline[i]
        v = delta + discount[i] * lmd * v
        value.insert(0, v + baseline[i])
    value = torch.stack(value)
    assert value.size() == reward.size(), value.size()
    return value


def split(trajectory, length):
    return [trajectory[i:i + length] for i in range(0, len(trajectory), length)]


def terminals(trajectories):
    return [t[0].baseline.detach().item() for t in trajectories[1:]] + [0]


@wrap.problem.truncation
@_wrap.checkpoint
@_wrap.opponent.eval
@_wrap.opponent.train
@_wrap.evaluate
@_wrap.optimizer
@_wrap.agent
@_wrap.model
@wrap.problem.context
@pg.wrap.problem.context
@_wrap.problem
@wrap.gae
@wrap.attr
@pg.wrap.problem.batch_size
@pg.wrap.agent
@pg.wrap.attr
class RL(object):
    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def close(self):
        pass

    def __len__(self):
        return 1

    def rollout(self):
        opponent = self.get_opponent_train_agent()
        controllers, ticks = self.problem.reset(self.kind, *opponent)
        trajectory = asyncio.get_event_loop().run_until_complete(asyncio.gather(
            wuji.problem.mdp.rollout(controllers[0], self.agent),
            *[wuji.problem.mdp._rollout(controller, agent) for controller, agent in zip(controllers[1:], opponent.values())],
            *map(wuji.problem.mdp.ticking, ticks),
        ))[0]
        return trajectory, torch.zeros([]), controllers[0].get_result()

    def value(self, **kwargs):
        return cumulate(kwargs['reward'], kwargs['discount'], kwargs['terminal'])

    def get_tensors(self, trajectory, terminal):
        logits = torch.cat([exp['logits'] for exp in trajectory])
        prob = torch.cat([exp['prob'] for exp in trajectory])
        baseline = torch.cat([exp['baseline'] for exp in trajectory]).view(-1)
        action = torch.cat([exp['action'] for exp in trajectory])
        reward = np.array([exp['reward'] for exp in trajectory])
        reward = self.norm(reward)
        reward = torch.FloatTensor(reward)
        discount = torch.FloatTensor(np.array([0 if exp['done'] else self.discount for exp in trajectory]))
        with torch.no_grad():
            value = self.value(baseline=baseline, reward=reward, discount=discount, terminal=terminal)
        return logits, prob, baseline, action, value

    def __next__(self):
        trajectory, terminal, result = self.rollout()
        return sum(exp.get('cost', 1) for exp in trajectory), self.get_tensors(trajectory, terminal), result

    def get_losses(self, logits, prob, baseline, action, value):
        advantage = value - baseline.detach()
        xe = self.agent.cross_entropy(logits=logits, prob=prob, action=action)
        return Losses(
            (advantage * xe).mean(),
            self.loss_critic(baseline, value),
            (-torch.log(prob) * prob).sum(-1).mean(),
        )

    def total_loss(self, policy, critic, entropy):
        return (torch.stack([policy, critic, -entropy]) * self.weight_loss).sum()

    def backward(self):
        cost, tensors, result = next(self)
        losses = self.get_losses(*tensors)
        loss = self.total_loss(*losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad_norm()
        return Outcome(cost, loss, losses, result)

    def __call__(self):
        outcome = self.backward()
        self.optimizer.step()
        return outcome
