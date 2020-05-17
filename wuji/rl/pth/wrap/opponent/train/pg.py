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
import traceback

from wuji.rl.pth.wrap.opponent import NAME as _NAME

NAME = '_'.join([
    _NAME + 's',
    os.path.basename(os.path.dirname(__file__)),
])


def prob_min(rl):
    name = '_'.join([
        inspect.getframeinfo(inspect.currentframe()).function,
        _NAME,
    ])

    def update(self):
        prob_min = self.get_prob_min_opponent()
        try:
            for opponent in getattr(self, NAME).agent:
                for agent in opponent.values():
                    agent.set_prob_min(prob_min)
        except AttributeError:
            traceback.print_exc()

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, name)
            self.set_prob_min_opponent(0)
            # self.set_prob_min_opponent(self.agent.prob_min)

        def set_prob_min_opponent(self, value):
            setattr(self, name, value)
            update(self)

        def get_prob_min_opponent(self):
            return getattr(self, name)

        def set_opponents_train(self, *args, **kwargs):
            try:
                return super().set_opponents_train(*args, **kwargs)
            finally:
                update(self)
    return RL
