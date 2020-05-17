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
import functools
import itertools
import operator
import warnings

import ray

import wuji
from . import problem, call, hook

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def attr(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.discount = self.config.getfloat('rl', 'discount')
            self.batch_size = self.config.getint('train', 'batch_size')
            self.rho = self.config.getfloat(NAME, 'rho')
            self.c = self.config.getfloat(NAME, 'c')
            assert self.rho >= self.c, (self.rho, self.c)
    return RL


def task(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task = []

        def __next__(self):
            if self.task:
                assert len(self.task) == len(self), (len(self.task), len(self))
            else:
                for actor, async_call in zip(self.actor, self.async_call):
                    for key, (args, kwargs) in async_call.items():
                        getattr(actor, key).remote(*args, **kwargs)
                self.async_call = [{} for _ in range(len(self.actor))]
                self.task = [actor._rollout.remote() for actor in self.actor]
            return super().__next__()
    return RL


def evaluate(rl):
    def idle_actor(self):
        if self.task:
            (ready,), _ = ray.wait(self.task)
            return self.actor[self.task.index(ready)]
        else:
            return self.actor[0]

    class RL(rl):
        def evaluate(self):
            sample = self.config.getint('sample', 'eval')
            _blob = ray.put(self.model.get_blob())
            ray.get([actor.set_blob.remote(_blob) for actor in self.actor])
            opponents = self.get_opponents_eval()
            if sample < len(opponents):
                warnings.warn(f'sample={sample} is lesser than the number of opponents ({len(opponents)})')
            results = ray.get(wuji.ray.submit(self.actor, [functools.partial(lambda actor, args: actor.evaluate_map.remote(*args), args=args) for args in enumerate(itertools.islice(itertools.cycle(opponents), sample))]))
            costs, results = zip(*[items[1:] for items in sorted(results, key=operator.itemgetter(0))])
            result = ray.get(idle_actor(self).evaluate_reduce.remote(ray.put(results)))
            digest = self.get_opponents_eval_digest()
            if digest:
                result['digest_opponents_eval'] = digest
            return sum(costs), result
    return RL
