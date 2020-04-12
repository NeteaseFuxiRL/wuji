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

import contextlib
import asyncio
import types

import numpy as np
import torch


async def ticking(tick):
    while not await tick():
        pass


async def rollout(controller, agent, step=np.iinfo(np.int).max, message=False):
    assert step > 0, step
    state = controller.get_state()
    trajectory = []
    for _ in range(step):
        exp = agent(state)
        exp['state'] = state
        action = agent.numpy(exp['action'])[0]
        exp.update(await controller(action))
        exp['state_'] = state = controller.get_state()
        controller.update(exp)
        if message:
            exp['message'] = repr(controller)
        trajectory.append(exp)
        if exp['done']:
            break
    return trajectory


async def _rollout(controller, agent, step=np.iinfo(np.int).max, message=False):
    assert step > 0, step
    state = controller.get_state()
    cost = 0
    for _ in range(step):
        with torch.no_grad():
            exp = agent(state)
        action = agent.numpy(exp['action'])[0]
        exp = await controller(action)
        state = controller.get_state()
        cost += exp.get('cost', 1)
        if message:
            exp['message'] = repr(controller)
        if exp['done']:
            break
    return cost


def evaluate(problem, kind, agent, opponents, loop=asyncio.get_event_loop()):
    cost = 0
    results = []
    for seed, opponent in enumerate(opponents):
        with contextlib.closing(problem.evaluating(seed)):
            controllers, ticks = problem.reset(kind, *opponent)
            costs = loop.run_until_complete(asyncio.gather(
                _rollout(controllers[0], agent),
                *[_rollout(controller, agent) for controller, agent in zip(controllers[1:], opponent.values())],
                *map(ticking, ticks),
            ))[:len(controllers)]
        cost += max(costs)
        results.append(controllers[0].get_result())
    return cost, problem.evaluate_reduce(results)


def wait(task, tasks, loop=asyncio.get_event_loop()):
    while True:
        done, pending = loop.run_until_complete(asyncio.wait({task} | tasks, return_when=asyncio.FIRST_COMPLETED))
        for t in done:
            if t not in tasks:
                return t, pending


class Truncator(object):
    def __init__(self, problem, kind, get_agent, get_opponent, step, loop=asyncio.get_event_loop()):
        self.problem = problem
        self.kind = kind
        self.get_agent = get_agent
        self.get_opponent = get_opponent
        self.step = step
        self.loop = loop
        self.done = True

    def reset(self):
        assert self.done
        self.done = False
        opponent = self.get_opponent()
        controllers, ticks = self.problem.reset(self.kind, *opponent)
        self.controller = controllers[0]
        self.tasks = {asyncio.Task(_rollout(controller, agent)) for controller, agent in zip(controllers[1:], opponent.values())} | {asyncio.Task(ticking(tick)) for tick in ticks}

    def __next__(self):
        assert self.step > 0, self.step
        if self.done:
            self.reset()
        agent = self.get_agent()
        task, pending = wait(asyncio.Task(rollout(self.controller, agent, step=self.step)), self.tasks, loop=self.loop)
        trajectory = task.result()
        if trajectory[-1]['done']:
            self.loop.run_until_complete(asyncio.gather(*pending))
            self.done = True
        return trajectory


def freq(outputs, trajectory, default=True):
    freq = np.zeros([outputs, 2], np.int)
    for exp in trajectory:
        if any(exp['state'].get('legal', [True])):
            freq[exp['action'], int(exp.get('success', default))] += 1
    return freq
