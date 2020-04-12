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
import types
import numbers
import shutil
import tarfile

from tensorboardX import SummaryWriter

import wuji.record
import wuji.recorder
from . import record


class Reset(object):
    def __init__(self, root_log, ext='.tar.gz'):
        self.root_log = root_log
        self.ext = ext

    def __call__(self, recorder):
        root_log = recorder.writer.logdir
        recorder.writer.close()
        with tarfile.open(root_log + self.ext, 'w:gz') as tar:
            tar.add(root_log, arcname=os.path.basename(root_log))
        shutil.rmtree(root_log, ignore_errors=True)
        wuji.file.tidy(os.path.dirname(root_log))
        os.makedirs(self.root_log, exist_ok=True)
        recorder.writer = SummaryWriter(self.root_log)


def training(evaluator):
    class Training(object):
        def __init__(self, evaluator):
            self.evaluator = evaluator
            evaluator.recorder.put = evaluator.recorder._put
            if 'index' in evaluator.kwargs:
                evaluator.recorder.put(Reset(evaluator.kwargs['root_log']))

        def close(self):
            self.evaluator.recorder.put = lambda *args, **kwargs: None

    class Evaluator(evaluator):
        def create_recorder(self):
            recorder = super().create_recorder()
            assert not hasattr(recorder, '_put')
            recorder._put = recorder.put
            recorder.put = lambda *args, **kwargs: None
            return recorder

        def training(self):
            training = []
            if not isinstance(self.recorder, wuji.recorder.Fake):
                training.append(Training(self))
            if hasattr(super(), 'training'):
                training.append(super().training())

            def close():
                for t in training[::-1]:
                    t.close()
            return types.SimpleNamespace(close=close)

        def evaluate(self):
            result = super().evaluate()
            self.recorder.put(wuji.record.Scalar(self.cost, **{'evaluate/' + key: value for key, value in result.items() if isinstance(value, (numbers.Integral, numbers.Real)) and not key.startswith('_')}))
            return result
    return Evaluator
