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

import numpy as np

from ... import blockmaze


def length(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, name)
                setattr(self, name, 0)

            async def __call__(self, *args, **kwargs):
                exp = await super().__call__(*args, **kwargs)
                setattr(self, name, getattr(self, name) + 1)
                return exp

            def get_result(self):
                result = super().get_result()
                result[name] = getattr(self, name)
                result['objective'].append(result[name])
                result['point'].append(result[name])
                return result
    return Problem


def final_distance(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_result(self):
                result = super().get_result()
                result[name] = np.sqrt(np.sum((np.array(self.env.maze.objects.agent.positions[0]) - np.array(blockmaze.goal_idx[0])) ** 2))
                result['fitness'] = -result[name]
                result['objective'].append(result['fitness'])
                result['point'].append(result['fitness'])
                return result
    return Problem


def position_set(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, name)
                setattr(self, name, set())

            def get_state(self):
                state = super().get_state()
                position, = self.env.maze.objects.agent.positions
                getattr(self, name).add(tuple(position))
                return state

            def get_result(self):
                result = super().get_result()
                result[name] = len(getattr(self, name))
                result['objective'].append(result[name])
                result['point'].append(result[name])
                return result
    return Problem
