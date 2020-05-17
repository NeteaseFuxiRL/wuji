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
import collections

import torch.cuda

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def _entropy(rl):
    name = inspect.getframeinfo(inspect.currentframe()).function

    class RL(rl):
        Loss = collections.namedtuple('Loss', rl.Loss._fields + (name,))

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.weight_loss = torch.cat([self.weight_loss, torch.FloatTensor([config.getfloat(NAME + '_weight_loss', name)])])

        def get_loss(self, logits, prob, baseline, action, value):
            entropy = (-torch.log(prob) * prob).sum(-1)
            return RL.Loss(*super().get_loss(logits, prob, baseline, action, value), entropy.mean())

        def total_loss(self):
            return (torch.stack([self.loss.policy, self.loss.critic, -self.loss.entropy]) * self.weight_loss).sum()
    return RL
