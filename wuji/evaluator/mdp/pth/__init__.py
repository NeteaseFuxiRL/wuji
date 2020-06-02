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

from wuji.rl.pth import wrap as wrap_rl
from .. import Evaluator as _Evaluator, wrap as _wrap


@wrap_rl.model
@wrap_rl.problem
@_wrap.evaluate
class Evaluator(_Evaluator):
    @staticmethod
    def ray_resources(config):
        return dict(num_cpus=1)

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def update_context(self, context):
        context['encoding']['blob']['module'] = self.config.get('model', 'module').split() + self.config.get('model', 'init').split()
        context['encoding']['blob']['agent'] = dict(eval=['.'.join([pg.agent.__name__, 'Eval'])])
