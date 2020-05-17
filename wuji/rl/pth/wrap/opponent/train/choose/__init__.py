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
import hashlib
import itertools


def random(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    import random

    class RL(rl):
        def choose_opponent_train(self):
            opponents = self.get_opponents_train()
            return random.choice(opponents)
    return RL


def cycle(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class RL(rl):
        def set_opponents_train(self, opponents):
            try:
                return super().set_opponents_train(opponents)
            finally:
                setattr(self, _name, itertools.cycle(self.get_opponents_train()))

        def choose_opponent_train(self):
            attr = getattr(self, _name)
            return next(attr)
    return RL


def partition(prob=0.5):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    import random

    def decorate(rl):
        class RL(rl):
            def choose_opponent_train(self):
                opponents = list(self.get_opponents_train())
                if random.random() < prob:
                    return opponents[-1]
                else:
                    opponents_former = opponents[:-1]
                    if opponents_former:
                        return random.choice(opponents_former)
                    else:
                        return opponents[-1]
        return RL
    return decorate
