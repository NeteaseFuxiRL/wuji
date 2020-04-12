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


def reward(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_rewards(self, **kwargs):
                rewards = super().get_rewards(**kwargs)
                rewards = rewards * self.problem._weight_reward
                return rewards

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, '_weight_reward')
            self._weight_reward = np.ones(len(self.context['weight_reward']))

        def get_hparam_real(self):
            def make(i):
                return dict(
                    boundary=np.array([(0, 1)], np.float),
                    set=lambda value: self._weight_reward.__setitem__(i, float(value)),
                    get=lambda: np.array([self._weight_reward[i]], np.float),
                )
            hparam = super().hparam_real() if hasattr(super(Problem, self), 'hparam_real') else {}
            for i, key in enumerate(self.context['name_reward']):
                if self.context['weight_reward'][i] > 0:
                    hparam[f'weight_reward/{key}'] = make(i)
            return hparam
    return Problem


def final_reward(problem):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Problem(problem):
        class Controller(problem.Controller):
            def get_rewards(self, **kwargs):
                rewards = super().get_rewards(**kwargs)
                rewards = rewards * self.problem._weight_final_reward
                return rewards

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, '_weight_final_reward')
            self._weight_final_reward = np.ones(len(self.context['weight_final_reward']))

        def get_hparam_real(self):
            def make(i):
                return dict(
                    boundary=np.array([(0, 1)], np.float),
                    set=lambda value: self._weight_final_reward.__setitem__(i, float(value)),
                    get=lambda: np.array([self._weight_final_reward[i]], np.float),
                )
            hparam = super().hparam_real() if hasattr(super(Problem, self), 'hparam_real') else {}
            for i, key in enumerate(self.context['name_final_reward']):
                if self.context['weight_final_reward'][i] > 0:
                    hparam[f'weight_final_reward/{key}'] = make(i)
            return hparam
    return Problem
