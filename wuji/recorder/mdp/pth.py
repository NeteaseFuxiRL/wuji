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

import traceback
import inspect
import functools
import asyncio

import wuji
import wuji.problem.mdp


def model(recorder):
    __wrapper__ = inspect.stack()[0][3]

    class Recorder(recorder):
        def create(self):
            import torch  # PyTorch's bug
            torch.set_num_threads(1)
            super().create()
            try:
                encoding = self.context['encoding']['blob']
                module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
                # TODO: kind
                kind = self.kwargs.get('kind', 0)
                self.model = module(self.config_test, **encoding['init'][kind]['kwargs'])
                self.model.eval()
                agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, encoding['agent']['eval']))(self.model)
                loop = asyncio.get_event_loop()
                (controller,), ticks = self.problem.reset(kind)
                trajectory = loop.run_until_complete(asyncio.gather(
                    wuji.problem.mdp.rollout(controller, agent),
                    *map(wuji.problem.mdp.ticking, ticks),
                ))[0]
                self.writer.add_graph(self.model, trajectory[0]['inputs'])
            except:
                traceback.print_exc()

        def __repr__(self):
            return f'{self.__module__}.{__wrapper__}<{super().__repr__()}>'
    return Recorder
