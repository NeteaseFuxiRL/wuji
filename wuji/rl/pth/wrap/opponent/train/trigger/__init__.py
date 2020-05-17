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
import functools
import types

import wuji


def stopper(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, _name)
            stopper = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.config.get('opponent_train', 'stopper').split('\t')))
            make = lambda: stopper(self)
            setattr(self, _name, types.SimpleNamespace(
                make=make,
                stopper=make(),
            ))

        def __call__(self, *args, **kwargs):
            outcome = super().__call__(*args, **kwargs)
            attr = getattr(self, _name)
            if attr.stopper(outcome):
                attr.stopper = attr.make()
                self.set_opponent_train(self.choose_opponent_train())
            return outcome
    return RL
