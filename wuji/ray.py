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
import configparser

import ray
import humanfriendly
import getpass

import wuji


def init(config, binary=False, **kwargs):
    def parse(key, value):
        if key in {'memory', 'object_store_memory'}:
            return humanfriendly.parse_size(value, binary=binary)
        elif key.endswith('port'):
            return int(value)
        else:
            return value
    try:
        kwargs = {**{key: parse(key, value) for key, value in config.items('ray')}, **kwargs}
    except configparser.NoSectionError:
        kwargs = {}
    if 'temp_dir' not in kwargs:
        kwargs['temp_dir'] = os.path.join(os.path.sep + 'tmp', 'ray', getpass.getuser())
    if 'address' in kwargs or 'redis_address' in kwargs:
        for key in 'memory object_store_memory temp_dir plasma_store_socket_name raylet_socket_name'.split():
            try:
                del kwargs[key]
            except KeyError:
                pass
    return kwargs


def submit(actors, pending):
    pending = iter(pending)
    running = [creator(actor) for actor, creator in zip(actors, pending)]
    done = []
    for task in pending:
        (ready,), _ = ray.wait(running)
        index = running.index(ready)
        actor = actors[index]
        running[index] = task(actor)
        done.append(ready)
    return done + running
