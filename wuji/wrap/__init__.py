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


def iteration(optimizer):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class Optimizer(optimizer):
        def __init__(self, *args, **kwargs):
            setattr(self, name, 0)
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            try:
                return super().__call__(*args, **kwargs)
            finally:
                setattr(self, name, getattr(self, name) + 1)
    return Optimizer
