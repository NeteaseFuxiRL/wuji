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
import configparser

import numpy as np
import torch.cuda
import torch.distributions
import torch.nn.functional as F

import wuji.agent.pth

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


class Eval(wuji.agent.pth.Model):
    def prob(self, logits, **kwargs):
        return F.softmax(logits, -1)

    def __call__(self, state):
        inputs = tuple(self.tensor(a.astype(np.float32), expand=0) for a in state['inputs'])
        logits = self.model(*inputs)
        prob = self.prob(logits, **state)
        _, action = prob.max(-1)
        return dict(
            inputs=inputs,
            logits=logits,
            prob=prob,
            action=action,
        )


class Train(Eval):
    def __init__(self, model, prob_min=None):
        self.config = model.config
        super().__init__(model)
        if prob_min is None:
            try:
                prob_min = self.config.getfloat(NAME, 'prob_min')
            except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
                prob_min = 0
        self.set_prob_min(prob_min)
        try:
            self.explore = getattr(self, 'explore_' + self.config.get(NAME, 'explore'))
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError, AttributeError):
            pass

    def set_prob_min(self, prob_min):
        assert 0 <= prob_min < 1, prob_min
        self.prob_min = float(prob_min)

    def prob(self, logits, **kwargs):
        prob = super().prob(logits, **kwargs)
        if self.prob_min > 0:
            return (prob + self.prob_min) / (1 + self.prob_min * prob.size(-1))
        else:
            return prob

    def cross_entropy(self, **kwargs):
        if self.prob_min > 0:
            policy = kwargs['prob'].gather(-1, kwargs['action'].view(-1, 1)).view(-1)
            return -torch.log(policy)
        else:
            return F.cross_entropy(kwargs['logits'], kwargs['action'], reduce=False)

    def explore(self, prob):
        return torch.distributions.Categorical(prob).sample()

    def explore_numpy(self, prob):
        return torch.LongTensor([np.random.choice(list(range(prob.size(-1))), p=prob[0].detach().cpu().numpy())])

    def __call__(self, state):
        inputs = tuple(self.tensor(a.astype(np.float32), expand=0) for a in state['inputs'])
        logits = self.model(*inputs)
        prob = self.prob(logits, **state)
        action = self.explore(prob)
        return dict(
            inputs=inputs,
            logits=logits,
            prob=prob,
            action=action,
        )
