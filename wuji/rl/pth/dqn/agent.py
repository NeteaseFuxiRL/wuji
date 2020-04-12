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
import random
import configparser

import numpy as np
import torch.cuda

import wuji.agent.pth

NAME = os.path.basename(os.path.dirname(__file__))


class Eval(wuji.agent.pth.Model):
    def __call__(self, state):
        inputs = tuple(self.tensor(a.astype(np.float32), expand=0) for a in state['inputs'])
        logits = self.model(*inputs)
        value, action = logits.max(-1)
        return dict(
            inputs=inputs,
            logits=logits,
            value=value,
            action=action,
        )


class Train(Eval):
    def __init__(self, model):
        super().__init__(model)
        self.epsilon = model.config.getfloat(NAME, 'epsilon')
        try:
            next_epsilon = eval('lambda epsilon: ' + model.config.get(NAME, 'next_epsilon'))
            self.next_epsilon = lambda: next_epsilon(self.epsilon)
        except configparser.NoOptionError:
            self.next_epsilon = lambda: self.epsilon

    def _get_outputs(self, *inputs):
        try:
            return self.outputs
        except AttributeError:
            logits = self.model(*inputs)
            self.outputs = logits.size(-1)
            return self.outputs

    def __call__(self, state):
        if random.random() < self.epsilon:
            inputs = tuple(self.tensor(a.astype(np.float32), expand=0) for a in state['inputs'])
            action = torch.randint(self._get_outputs(*inputs), [1])
            return dict(inputs=inputs, action=action)
        else:
            return super().__call__(state)

    def update(self):
        self.epsilon = self.next_epsilon()
