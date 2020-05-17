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
import types


def make_tensors(rl):
    name = '_'.join([
        os.path.basename(os.path.splitext(__file__)[0]),
        inspect.getframeinfo(inspect.currentframe()).function,
    ])

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, name)

        def __call__(self, *args, **kwargs):
            setattr(self, name, [])
            return super().__call__(*args, **kwargs)

        def make_tensors(self, *args, **kwargs):
            tensors = super().make_tensors(*args, **kwargs)
            attr = getattr(self, name)
            attr.append(types.SimpleNamespace(
                inputs=args,
                outputs=tensors,
            ))
            return tensors
    return RL
