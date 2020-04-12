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

import types
import functools


def patch_call(instance, func):
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
           return func(*arg, **kwarg)
    try:
        return types.SimpleNamespace(close=functools.partial(setattr, instance, '__class__', instance.__class__))
    finally:
        instance.__class__ = _


def cast_nonlocal(instance):
    for base in type(instance).mro():
        if '.<locals>.' not in str(base):
            try:
                return types.SimpleNamespace(close=functools.partial(setattr, instance, '__class__', instance.__class__))
            finally:
                instance.__class__ = base


def attr(instance, *args, **kwargs):
    class Restore(dict):
        def close(self):
            for key, value in self.items():
                setattr(instance, key, value)
    restore = Restore()
    for key in args:
        if hasattr(instance, key):
            assert key not in restore, key
            restore[key] = getattr(instance, key)
            delattr(instance, key)
    for key, value in kwargs.items():
        assert key not in restore, key
        restore[key] = getattr(instance, key)
        setattr(instance, key, value)
    return restore
