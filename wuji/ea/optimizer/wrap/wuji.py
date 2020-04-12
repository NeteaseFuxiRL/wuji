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

import wuji.record


def bug_set(log=False):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def decorate(optimizer):
        class Optimizer(optimizer):
            def __init__(self, *args, **kwargs):
                setattr(self, name, set())
                super().__init__(*args, **kwargs)

            def __getstate__(self):
                state = super().__getstate__()
                assert name not in state
                state[name] = getattr(self, name)
                return state

            def __setstate__(self, state):
                setattr(self, name, state.pop(name))
                return super().__setstate__(state)

            def __call__(self, *args, **kwargs):
                attr = getattr(self, name)
                try:
                    super().__call__(*args, **kwargs)
                finally:
                    num = len(attr)
                    for individual in self.offspring:
                        attr = attr.union(individual['result'][name])
                    assert len(attr) >= num, (num, attr)
                    setattr(self, name, attr)
                    if log and len(attr) > num:
                        self.recorder.put(wuji.record.Scalar(self.cost, bugs=len(attr)))
        return Optimizer
    return decorate
