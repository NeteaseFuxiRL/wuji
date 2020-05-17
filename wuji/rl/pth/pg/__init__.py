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

import numpy as np
import torch
import recordtype

import wuji.problem.mdp
import wuji.model.pth.wrap
from .. import wrap as _wrap
from . import wrap, agent

NAME = os.path.basename(os.path.dirname(__file__))
Outcome = recordtype.recordtype('Outcome', ['cost', 'loss', 'result'])


def cumulate(reward, discount, terminal=0):
    value = []
    v = terminal
    for r in reversed(torch.unbind(reward)):
        v = v * discount + r
        value.insert(0, v)
    value = torch.stack(value)
    assert value.size() == reward.size(), value.size()
    return value


def clip_reward_final(reward):
    _reward = reward[:-1]
    try:
        r = np.clip(reward[-1], _reward.min(), _reward.max())
    except:
        r = 0
    reward[-1] = r
    return reward


@_wrap.checkpoint
@_wrap.opponent.eval
@_wrap.opponent.train
@_wrap.evaluate
@_wrap.optimizer
@_wrap.agent
@_wrap.model
@wrap.problem.context
@_wrap.problem
@wrap.problem.batch_size
@wrap.agent
@wrap.attr
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
        return trajectory, controllers[0].get_result()

    def get_tensors(self, trajectory):
        logits = torch.cat([exp['logits'] for exp in trajectory])
        prob = torch.cat([exp['prob'] for exp in trajectory])
        action = torch.cat([exp['action'] for exp in trajectory])
        reward = np.array([exp['reward'] for exp in trajectory])
        reward = self.norm(reward)
        reward = torch.FloatTensor(reward)
        value = cumulate(reward, self.discount)
        return logits, prob, action, value

    def __next__(self):
        trajectory, result = self.rollout()
        return sum(exp.get('cost', 1) for exp in trajectory), self.get_tensors(trajectory), result

    def __call__(self):
        cost, (logits, prob, action, value), result = next(self)
        xe = self.agent.cross_entropy(logits=logits, prob=prob, action=action)
        loss = (value * xe).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad_norm()
        self.optimizer.step()
        return Outcome(cost, loss, result)
