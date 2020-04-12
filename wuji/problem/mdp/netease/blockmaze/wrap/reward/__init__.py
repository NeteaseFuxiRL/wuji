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
import configparser

import numpy as np


def goal(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_final_rewards(self, **kwargs):
                return np.append(super().get_final_rewards(**kwargs), 1 if self.info['goal'] else 0)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_name_final_reward(self.get_name_final_reward() + [name])
            self.set_weight_final_reward(np.append(self.get_weight_final_reward(), self.config.getfloat('blockmaze_weight_final_reward', name)))
    return Problem


def valid(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_rewards(self, **kwargs):
                return np.append(super().get_rewards(**kwargs), 1 if self.info['valid'] else 0)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_name_reward(self.get_name_reward() + [name])
            self.set_weight_reward(np.append(self.get_weight_reward(), self.config.getfloat('blockmaze_weight_reward', name)))
            try:
                lower, upper = map(float, self.config.get('blockmaze_weight_reward_boundary', name).split())
                self.context['weight_reward_boundary'] = {**self.context.get('weight_reward_boundary', {}), **{name: (lower, upper)}}
            except (configparser.NoSectionError, configparser.NoOptionError):
                pass
    return Problem


def invalid(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_rewards(self, **kwargs):
                return np.append(super().get_rewards(**kwargs), 0 if self.info['valid'] else -1)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_name_reward(self.get_name_reward() + [name])
            self.set_weight_reward(np.append(self.get_weight_reward(), self.config.getfloat('blockmaze_weight_reward', name)))
    return Problem


def length(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_rewards(self, **kwargs):
                return np.append(super().get_rewards(**kwargs), 1)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_name_reward(self.get_name_reward() + [name])
            self.set_weight_reward(np.append(self.get_weight_reward(), self.config.getfloat('blockmaze_weight_reward', name)))
    return Problem


def state_set(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_rewards(self, **kwargs):
                return np.append(super().get_rewards(**kwargs), len(self.env.state_per_traj - getattr(self, name)))

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_name_reward(self.get_name_reward() + [name])
            self.set_weight_reward(np.append(self.get_weight_reward(), self.config.getfloat('blockmaze_weight_reward', name)))
    return Problem
