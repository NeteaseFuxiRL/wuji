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
import functools
import copy

import wuji
import wuji.recorder


class Evaluator(object):
    def __init__(self, config, context, **kwargs):
        self.config = config
        self.context = context
        self.kwargs = kwargs
        if 'root' not in kwargs:
            kwargs['root'] = os.path.expanduser(os.path.expandvars(config.get('model', 'root')))
        if 'log' not in kwargs:
            kwargs['log'] = wuji.config.digest(config)
        if 'root_log' not in kwargs:
            kwargs['root_log'] = os.path.join(kwargs['root'], 'log', kwargs['log'])
        self.cost = 0
        if 'index' not in kwargs or kwargs['index'] < self.config.getint('train', 'record'):
            self.recorder = self.create_recorder()
            self.recorder.start()
        else:
            self.recorder = wuji.recorder.Fake(self.config, self.context, **self.kwargs)

    def close(self):
        return self.recorder.close()

    def create_recorder(self):
        recorder = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.config.get('record', 'recorder').split('\t')))
        return recorder(self.config, self.context, cost=self.cost, **copy.deepcopy(self.kwargs))

    def get_context(self):
        return self.context

    def __getstate__(self):
        state = {key: value for key, value in self.__dict__.items() if key in {'context', 'cost'}}
        state['decision'] = self.get()
        return state

    def __setstate__(self, state):
        self.set(state.pop('decision'))
        state.pop('context', None)
        self.cost = state.pop('cost', 0)

    def initialize(self):
        return {coding: getattr(self, 'initialize_' + coding)() for coding in self.context['encoding']}

    def set(self, decision):
        for coding in self.context['encoding']:
            if 'boundary' in self.context['encoding'][coding]:
                value = getattr(self, 'set_' + coding)(iter(decision[coding]))
                assert next(value, None) is None, self.context['encoding'][coding]['header']
            else:
                getattr(self, 'set_' + coding)(decision[coding])

    def get(self):
        return {coding: getattr(self, 'get_' + coding)() for coding in self.context['encoding']}
