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

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def attr(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.discount = self.config.getfloat('rl', 'discount')
            self.batch_size = self.config.getint('train', 'batch_size')
            self.epsilon = self.config.getfloat(NAME, 'epsilon')
            self.epoch = self.config.getint(NAME, 'epoch')
    return RL
