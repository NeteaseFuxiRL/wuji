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

import itertools
import re


def rsplit(n):
    def decorate(module):
        class Module(module):
            def __init__(self, *args, **kwargs):
                assert not hasattr(module, 'group')
                super().__init__(*args, **kwargs)

            def group(self):
                return [list(group) for prefix, group in itertools.groupby(self.state_dict().keys(), lambda key: key.rsplit('.', n)[0])]
        return Module
    return decorate


def last_digits(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            assert not hasattr(module, 'group')
            super().__init__(*args, **kwargs)

        def group(self):
            prog = re.compile('(^.*\.\d+)')

            def prefix(key):
                m = prog.search(key)
                if m is None:
                    return m
                else:
                    return m.group(0)
            return [list(group) for prefix, group in itertools.groupby(self.state_dict().keys(), prefix)]
    return Module
