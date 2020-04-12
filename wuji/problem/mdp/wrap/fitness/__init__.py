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


def cumulative(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, name)
                setattr(self, name, 0)

            def get_reward(self, *args, **kwargs):
                reward = super().get_reward(*args, **kwargs)
                setattr(self, name, getattr(self, name) + reward)
                return reward

            def get_result(self):
                result = super().get_result()
                result[name] = getattr(self, name)
                result['fitness'] = result[name]
                return result
    return Problem
