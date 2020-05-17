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

from .. import Evaluator


class RL(Evaluator):
    def __init__(self, config, rl, **kwargs):
        super().__init__(config, rl.context, **kwargs)
        self.rl = rl

    def close(self):
        return self.rl.close()

    def __getstate__(self):
        state = super().__getstate__()
        state.update(self.rl.__getstate__())
        return state

    def __setstate__(self, state):
        self.rl.__setstate__(state)
        return super().__setstate__(state)

    def __len__(self):
        return len(self.rl)

    def get_kind(self):
        return self.rl.kind

    def initialize_blob(self):
        return self.rl.initialize_blob()

    def set_blob(self, blob):
        self.rl.set_blob(blob)

    def get_blob(self):
        return self.rl.get_blob()

    def train(self):
        self.outcome = self.rl()
        self.cost += self.outcome.cost
        self.recorder()
        return self.outcome

    def evaluate(self):
        cost, result = self.rl.evaluate()
        self.cost += cost
        return result

    def set_opponent_train(self, *args, **kwargs):
        return self.rl.set_opponent_train(*args, **kwargs)

    def set_opponents_train(self, *args, **kwargs):
        return self.rl.set_opponents_train(*args, **kwargs)

    def set_opponents_eval(self, *args, **kwargs):
        return self.rl.set_opponents_eval(*args, **kwargs)
