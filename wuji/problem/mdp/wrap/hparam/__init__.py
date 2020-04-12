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

import numpy as np


def reward(problem):
    class Problem(problem):
        def get_hparam_real(self):
            def make(i, lower, upper):
                def set(value):
                    weight = self.get_weight_reward()
                    weight[i] = float(value)
                    return self.set_weight_reward(weight)

                return dict(
                    boundary=np.array([(lower, upper)], np.float),
                    set=set,
                    get=lambda: np.array([self.get_weight_reward()[i]], np.float),
                )
            try:
                hparam = super().get_hparam_real()
            except AttributeError:
                hparam = {}
            assert len(self.get_name_reward()) == len(self.get_weight_reward()), (self.get_name_reward(), len(self.get_weight_reward()))
            for i, (key, upper) in enumerate(zip(self.get_name_reward(), self.get_weight_reward())):
                lower, upper = self.context.get('weight_reward_boundary', {}).get(key, (0, upper))
                if upper < lower:
                    lower, upper = upper, lower
                if lower < upper:
                    hparam[f'weight_reward/{key}'] = make(i, lower, upper)
            return hparam
    return Problem


def final_reward(problem):
    class Problem(problem):
        def get_hparam_real(self):
            def make(i, lower, upper):
                def set(value):
                    weight = self.get_weight_final_reward()
                    weight[i] = float(value)
                    return self.set_weight_final_reward(weight)

                return dict(
                    boundary=np.array([(lower, upper)], np.float),
                    set=set,
                    get=lambda: np.array([self.get_weight_final_reward()[i]], np.float),
                )
            try:
                hparam = super().get_hparam_real()
            except AttributeError:
                hparam = {}
            assert len(self.get_name_final_reward()) == len(self.get_weight_final_reward()), (self.get_name_final_reward(), len(self.get_weight_final_reward()))
            for i, (key, upper) in enumerate(zip(self.get_name_final_reward(), self.get_weight_final_reward())):
                lower, upper = self.context.get('weight_final_reward_boundary', {}).get(key, (0, upper))
                if upper < lower:
                    lower, upper = upper, lower
                if lower < upper:
                    hparam[f'weight_final_reward/{key}'] = make(i, lower, upper)
            return hparam
    return Problem
