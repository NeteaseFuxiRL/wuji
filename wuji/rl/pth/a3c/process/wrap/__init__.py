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

import itertools
import operator
import functools
import inspect
import warnings

from . import call


def async_count(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        def __init__(self, *args, **kwargs):
            setattr(self, name, {})
            super().__init__(*args, **kwargs)

        def async_put(self, key, *args, **kwargs):
            attr = getattr(self, name)
            attr[key] = attr.get(key, 0) + 1
            return super().async_put(key, *args, **kwargs)

        def async_get(self, key):
            attr = getattr(self, name)
            assert attr[key] > 0, (key, attr[key])
            attr[key] -= 1
            return super().async_get(key)
    return RL


def async_reset(*names):
    def decorate(rl):
        def call(self, name, *args, **kwargs):
            for key, async_count in self.async_count.items():
                for _ in range(async_count):
                    self.async_get(key)
                self.async_count[key] = 0
            return getattr(super(rl, self), name)(*args, **kwargs)

        for name in names:
            setattr(rl, name, functools.partialmethod(call, name))
        return rl
    return decorate


def evaluate(rl):
    class RL(rl):
        def evaluate(self):
            sample = self.config.getint('sample', 'eval')
            blob = self.model.get_blob()
            self.broadcast('set_blob', blob)
            opponents = self.get_opponents_eval()
            if sample < len(opponents):
                warnings.warn(f'sample={sample} is lesser than the number of opponents ({len(opponents)})')
            for args in enumerate(itertools.islice(itertools.cycle(opponents), sample)):
                self.async_put('evaluate_map', *args)
            _, results = zip(*[self.async_get('evaluate_map') for _ in range(sample)])
            costs, results = zip(*[items[1:] for items in sorted(results, key=operator.itemgetter(0))])
            self.async_put('evaluate_reduce', results)
            index_actor, result = self.async_get('evaluate_reduce')
            digest = self.get_opponents_eval_digest()
            if digest:
                result['digest_opponents_eval'] = digest
            return sum(costs), result
    return RL


def opponent(rl):
    class RL(rl):
        def set_opponent_train(self, opponent):
            for kwargs in self.gradient_kwargs:
                kwargs['set_opponent_train'] = ((opponent,), {})
            return super().set_opponent_train(opponent)
    return RL
