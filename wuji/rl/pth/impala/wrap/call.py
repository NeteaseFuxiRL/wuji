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

import ray
from termcolor import colored


def any(*keys, force=False):
    name = inspect.getframeinfo(inspect.currentframe()).function

    def idle_actor(self):
        if self.task:
            (ready,), _ = ray.wait(self.task)
            return self.actor[self.task.index(ready)]
        else:
            return self.actor[0]

    def call(self, key, *args, **kwargs):
        actor = idle_actor(self)
        _args = tuple(map(ray.put, args))
        _kwargs = {key: ray.put(value) for key, value in kwargs.items()}
        return ray.get(getattr(actor, key).remote(*_args, **_kwargs))

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
        _args = tuple(map(ray.put, args))
        _kwargs = {key: ray.put(value) for key, value in kwargs.items()}
        return ray.get([getattr(actor, key).remote(*_args, **_kwargs) for actor in self.actor])[0]

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


def all_async(*keys):
    def decorate(rl):
        def call(self, func, key, *args, **kwargs):
            _args = tuple(map(ray.put, args))
            _kwargs = {key: ray.put(value) for key, value in kwargs.items()}
            for async_call in self.async_call:
                async_call[key] = (_args, _kwargs)
            return func(self, *args, **kwargs)
        for key in keys:
            setattr(rl, key, functools.partialmethod(call, getattr(rl, key), key))
        return rl
    return decorate
