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

import numpy as np

from .. import pg


class Eval(pg.agent.Eval):
    def __call__(self, state):
        inputs = tuple(self.tensor(a.astype(np.float32), expand=0) for a in state['inputs'])
        logits, baseline = self.model(*inputs)
        prob = self.prob(logits, **state)
        _, action = prob.max(-1)
        return dict(
            inputs=inputs,
            logits=logits,
            baseline=baseline,
            prob=prob,
            action=action,
        )


class Train(pg.agent.Train):
    def __call__(self, state):
        inputs = tuple(self.tensor(a.astype(np.float32), expand=0) for a in state['inputs'])
        logits, baseline = self.model(*inputs)
        prob = self.prob(logits, **state)
        action = self.explore(prob)
        return dict(
            inputs=inputs,
            logits=logits,
            baseline=baseline,
            prob=prob,
            action=action,
        )
