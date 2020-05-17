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
import functools

from termcolor import colored


def any(*keys, force=False):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def call(self, key, *args, **kwargs):
        self.async_put(key, *args, **kwargs)
        _, result = self.async_get(key)
        return result

    def decorate(rl):
        wraped = []
        unwraped = []
        for key in keys:
            if hasattr(rl, key) and not force:
                unwraped.append(key)
            else:
                wraped.append(key)
                setattr(rl, key, functools.partialmethod(call, key))
        print('\n'.join([
            f'call.{name}({rl}):',
            colored('\n'.join(['wraped:'] + [f'\t{key}' for key in wraped]), 'green'),
            colored('\n'.join(['unwraped:'] + [f'\t{key}' for key in unwraped]), 'red'),
        ]))
        return rl
    return decorate


def all(*keys, force=False):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def call(self, key, *args, **kwargs):
        return self.broadcast(key, *args, **kwargs)

    def decorate(rl):
        wraped = []
        unwraped = []
        for key in keys:
            if hasattr(rl, key) and not force:
                unwraped.append(key)
            else:
                wraped.append(key)
                setattr(rl, key, functools.partialmethod(call, key))
        print('\n'.join([
            f'call.{name}({rl}):',
            colored('\n'.join(['wraped:'] + [f'\t{key}' for key in wraped]), 'green'),
            colored('\n'.join(['unwraped:'] + [f'\t{key}' for key in unwraped]), 'red'),
        ]))
        return rl
    return decorate
