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
import itertools

import torch


def load(path):
    data = torch.load(path)
    try:
        blobs = [individual['decision']['blob'] for individual in data['population']]
        try:
            _, indexes = os.path.basename(os.path.splitext(path)[0]).split(':')
            indexes = list(map(int, indexes.split(',')))
            blobs = [blobs[index] for index in indexes]
        except ValueError:
            pass
    except KeyError:
        blobs = [data['decision']['blob']]
    return blobs


def file(kind=1):
    def decorate(evaluator):
        class Evaluator(evaluator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                root = os.path.expanduser(os.path.expandvars(self.config.get('opponent_file', str(kind))))
                blobs = list(itertools.chain(*(load(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(root) for filename in filenames if filename.endswith('.pth'))))
                assert blobs
                self.set_opponents_eval([{kind: blob} for blob in blobs])
        return Evaluator
    return decorate
