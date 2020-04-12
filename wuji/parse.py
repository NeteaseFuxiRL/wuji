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

import importlib
import re


def attr(s):
    try:
        module, name = s.rsplit('.', 1)
    except ValueError:
        return eval(s)
    module = importlib.import_module(module)
    return getattr(module, name)


def instance(s, **kwargs):
    r = re.match(r'([a-zA-Z0-9_.]+)(\.[a-zA-Z0-9_.]+\(.+)', s)
    if r is None:
        return attr(s)
    m = r.group(1)
    module = importlib.import_module(m)
    return eval('module' + r.group(2), {**globals(), **dict(module=module)}, kwargs)


def chain(s, resources):
    if not s:
        raise SyntaxError()
    chain = []
    for s in s.split():
        prefix, suffix = s.split(':')
        try:
            num = int(suffix)
        except ValueError:
            num = resources[suffix]
        chain += [prefix] * num
    return chain
