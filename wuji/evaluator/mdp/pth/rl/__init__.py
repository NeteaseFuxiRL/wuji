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

import types

import wuji
import wuji.recorder
import wuji.record.mdp
import wuji.evaluator.wrap as wrap
import wuji.evaluator.wrap.hparam as wrap_hparam
from ... import RL as _Evaluator


@wrap.training
@wrap.record.outcome.result
@wrap_hparam.decision(problem=lambda evaluator: evaluator.rl)
class Evaluator(_Evaluator):
    def training(self):
        header = {coding: encoding['header'] for coding, encoding in self.context['encoding'].items() if 'header' in encoding}
        if any(header.values()):
            self.recorder.put(wuji.record.Text(self.cost, **{
                'hparam': '\n\n'.join([coding + ': ' + ', '.join([name + '=' + str(value) for name, value in zip(header, getattr(self, 'get_' + coding)())]) for coding, header in header.items()]),
            }))
        return types.SimpleNamespace(close=lambda: None)
