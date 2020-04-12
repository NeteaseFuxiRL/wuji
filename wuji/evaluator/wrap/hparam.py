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
import random
import itertools
import collections

import numpy as np


def decision(**kwargs):
    name = os.path.basename(os.path.splitext(__file__)[0])

    def decorate(evaluator):
        class Problem(evaluator):
            def __init__(self, *_args, **_kwargs):
                super().__init__(*_args, **_kwargs)
                assert not hasattr(self, name)
                setattr(self, name, getattr(self, 'get_' + name)())
                if 'encoding' not in self.context:
                    self.context['encoding'] = {}
                for coding, hparam in getattr(self, name).items():
                    if hparam:
                        context = self.context['encoding'].get(coding, dict())
                        boundary = np.concatenate([hparam['boundary'] for hparam in hparam.values()])
                        if 'boundary' in context:
                            context['boundary'] = np.vstack([boundary, context['boundary']])
                        else:
                            context['boundary'] = boundary
                        context['header'] = list(itertools.chain(*[[f'{key}{i}' for i in range(len(hparam['boundary']))] if len(hparam['boundary']) > 1 else [key] for key, hparam in hparam.items()])) + context.get('header', [])
                        assert len(context['boundary']) == len(context['header']), (len(context['boundary']), len(context['header']))
                        self.context['encoding'][coding] = context
                for prefix, fetch in kwargs.items():
                    for coding, src in fetch(self).context.get('encoding', {}).items():
                        if hasattr(self.context['encoding'], coding):
                            dst = self.context['encoding'][coding]
                            dst['boundary'] = np.concatenate([dst['boundary'], src['boundary']])
                            dst['header'] += ['_'.join([prefix, header]) for header in src['header']]
                        else:
                            self.context['encoding'][coding] = src

            def get_hparam(self):
                return {method[11:]: getattr(self, method)() for method in dir(self) if method.startswith('get_hparam_')}

            def initialize_real(self):
                method = inspect.getframeinfo(inspect.currentframe()).function
                coding = method.rsplit('_', 1)[-1]
                boundary = [hparam['boundary'] for hparam in getattr(self, name).get(coding, {}).values()]
                if boundary:
                    lower, upper = np.concatenate(boundary).T
                    decision = lower + np.random.random(lower.shape) * (upper - lower)
                else:
                    decision = np.array([], np.float)
                for fetch in kwargs.values():
                    decision = np.concatenate([decision, getattr(fetch(self), method)()])
                return decision

            def set_real(self, value):
                assert isinstance(value, collections.Iterable), type(value)
                method = inspect.getframeinfo(inspect.currentframe()).function
                coding = method.rsplit('_', 1)[-1]
                for key, hparam in getattr(self, name).get(coding, {}).items():
                    hparam['set'](*[next(value) for _ in hparam['boundary']])
                for fetch in kwargs.values():
                    value = getattr(fetch(self), method)(value)
                return value

            def get_real(self):
                method = inspect.getframeinfo(inspect.currentframe()).function
                coding = method.rsplit('_', 1)[-1]
                decision = [hparam['get']() for hparam in getattr(self, name).get(coding, {}).values()]
                if decision:
                    decision = np.concatenate(decision)
                else:
                    decision = np.array([], np.float)
                for fetch in kwargs.values():
                    decision = np.concatenate([decision, getattr(fetch(self), method)()])
                return decision

            def initialize_integer(self):
                method = inspect.getframeinfo(inspect.currentframe()).function
                coding = method.rsplit('_', 1)[-1]
                boundary = [hparam['boundary'] for hparam in getattr(self, name).get(coding, {}).values()]
                if boundary:
                    lower, upper = np.concatenate(boundary).T
                    decision = np.array([random.randint(l, u) for l, u in zip(lower, upper)])
                else:
                    decision = np.array([], np.int)
                for fetch in kwargs.values():
                    decision = np.concatenate([decision, getattr(fetch(self), method)()])
                return decision

            def set_integer(self, value):
                assert isinstance(value, collections.Iterable), type(value)
                method = inspect.getframeinfo(inspect.currentframe()).function
                coding = method.rsplit('_', 1)[-1]
                for key, hparam in getattr(self, name).get(coding, {}).items():
                    hparam['set'](*[next(value) for _ in hparam['boundary']])
                for fetch in kwargs.values():
                    value = getattr(fetch(self), method)(value)
                return value

            def get_integer(self):
                method = inspect.getframeinfo(inspect.currentframe()).function
                coding = method.rsplit('_', 1)[-1]
                decision = [hparam['get']() for hparam in getattr(self, name).get(coding, {}).values()]
                if decision:
                    decision = np.concatenate(decision)
                else:
                    decision = np.array([], np.int)
                for fetch in kwargs.values():
                    decision = np.concatenate([decision, getattr(fetch(self), method)()])
                return decision
        return Problem
    return decorate
