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
import hashlib
import types
import itertools
import collections
import configparser
import logging


def check(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, self.config.getboolean('opponent_train', 'debug'))

        def set_opponent_train(self, opponent, *args, **kwargs):
            if getattr(self, _name):
                digest = {kind: hashlib.md5(bytes(itertools.chain(*[value.cpu().numpy().tostring() for value in blob.values()]))).hexdigest() for kind, blob in opponent.items()}
                logging.warning(f'set opponent_train={digest}')
            return super().set_opponent_train(opponent, *args, **kwargs)

        def get_opponent_train_agent(self):
            opponent = super().get_opponent_train_agent()
            assert opponent
            if getattr(self, _name):
                digest = {kind: hashlib.md5(bytes(itertools.chain(*[value.cpu().numpy().tostring() for value in agent.model.get_blob().values()]))).hexdigest() for kind, agent in opponent.items()}
                logging.warning(f'get opponent_train (agent)={digest}')
            return opponent
    return RL


def opponents(capacity='opponent_train/capacity'):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def decorate(rl):
        class RL(rl):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, _name)
                try:
                    maxlen = self.config.getint(*capacity.split('/')) if isinstance(capacity, str) else capacity
                    if maxlen > 0:
                        limit = lambda opponents: collections.deque(opponents, maxlen=maxlen)
                    else:
                        limit = lambda opponents: opponents
                except configparser.NoOptionError:
                    limit = lambda opponents: opponents
                setattr(self, _name, types.SimpleNamespace(limit=limit, opponents=[{}]))

            def set_opponents_train(self, opponents):
                assert isinstance(opponents, collections.abc.Iterable), type(opponents)
                assert opponents
                assert all(isinstance(opponent, dict) for opponent in opponents), [type(opponent) for opponent in opponents]
                attr = getattr(self, _name)
                attr.opponents = attr.limit(opponents)
                self.set_opponent_train(self.choose_opponent_train())

            def get_opponents_train(self):
                attr = getattr(self, _name)
                return attr.opponents
        return RL
    return decorate
