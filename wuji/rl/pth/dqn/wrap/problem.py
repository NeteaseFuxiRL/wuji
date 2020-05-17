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

from .. import agent

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def context(rl):
    class RL(rl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context['encoding']['blob']['module'] = self.config.get('model', 'module').split() + self.config.get('model', 'init').split()
            self.context['encoding']['blob']['agent'] = dict(
                train=['.'.join([agent.__name__, 'Train'])] + self.config.get(NAME, 'agent_train').split(),
                eval=['.'.join([agent.__name__, 'Eval'])] + self.config.get(NAME, 'agent_eval').split(),
            )
    return RL
