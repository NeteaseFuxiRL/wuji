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

import functools
import itertools

import torch

import wuji.problem.mdp


def evaluate(evaluator):
    class Evaluator(evaluator):
        def evaluate(self):
            with torch.no_grad():
                try:
                    self.model.eval()
                    agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, self.context['encoding']['blob']['agent']['eval']))(self.model)
                    cost, result = wuji.problem.mdp.evaluate(self.problem, self.kind, agent, itertools.repeat({}, self.problem.config.getint('sample', 'eval')))
                    self.cost += cost
                    return result
                finally:
                    self.model.train()
    return Evaluator
