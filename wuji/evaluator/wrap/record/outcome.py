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

import numpy as np

import wuji.record


def result(evaluator):
    name = inspect.getframeinfo(inspect.currentframe()).function
    _name = hashlib.md5((__file__ + name).encode()).hexdigest()

    def fetch(self):
        result = getattr(self, _name)
        setattr(self, _name, {})
        return {'outcome/result/' + key: value for key, value in result.items() if np.isscalar(value) and not key.startswith('_')}

    class Evaluator(evaluator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, _name)
            setattr(self, _name, {})

        def train(self, *args, **kwargs):
            outcome = super().train(*args, **kwargs)
            if outcome.result:
                setattr(self, _name, outcome.result)
            return outcome

        def create_recorder(self):
            recorder = super().create_recorder()
            recorder.register(self.config.get('record', 'scalar'), lambda: wuji.record.Scalar(self.cost, **fetch(self)))
            return recorder
    return Evaluator
