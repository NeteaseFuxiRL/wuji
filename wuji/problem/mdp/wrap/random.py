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

import logging

import wuji


def reset_seed(seed=0, before=True, after=False):
    def _seed(self):
        wuji.random.seed(seed, prefix=f'reset seed={seed}: ')
        if hasattr(self, 'seed'):
            self.seed(seed)
        else:
            logging.warning(f'{self} has no seed method')

    def decorate(problem):
        class Problem(problem):
            def reset(self, *args, **kwargs):
                if before:
                    _seed(self)
                try:
                    return super().reset(*args, **kwargs)
                finally:
                    if after:
                        _seed(self)
        return Problem
    return decorate
