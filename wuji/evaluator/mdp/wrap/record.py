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
import contextlib
import numbers
import asyncio
import functools
import itertools
import shutil
import configparser
import logging

import numpy as np
import torch
import tqdm

import wuji.temp
import wuji.problem.mdp
import wuji.record.mdp.image
from wuji.rl.pth.wrap.opponent import make_agents
from wuji.evaluator.mdp.wrap.opponent import load


class Rollout(object):
    def __init__(self, evaluator, evaluating=False, tag='rollout', **kwargs):
        self.cost = evaluator.cost
        self.context = evaluator.context
        self.kind = getattr(evaluator, 'kind', 0)
        self.blob = evaluator.get_blob()
        self.evaluating = evaluating
        self.tag = tag
        self.kwargs = kwargs

    def __call__(self, recorder):
        config = recorder.config_test
        problem = recorder.problem
        kind = self.kwargs.get('kind', 0)
        encoding = self.context['encoding']['blob']
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][kind]['kwargs'])
        model.set_blob(self.blob)
        agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, encoding['agent']['eval']))(model)
        try:
            kind, value = next(iter({int(key): value for key, value in config.items('opponent_file') if key.isdigit()}.items()))
            opponents = [make_agents(config, encoding, {kind: blob}) for blob in itertools.chain(*(load(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(os.path.expanduser(os.path.expandvars(value))) for filename in filenames))]
        except configparser.NoSectionError:
            opponents = [{}]
        init = problem.context['encoding']['blob']['init'][self.kind]
        root = os.path.join(recorder.kwargs['root'], 'trajectory', str(self.cost))
        logging.info(root)
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        with contextlib.closing(wuji.temp.cast_nonlocal(problem)):
            torch.save(problem, root + '.pth')
        freq = []
        results = []
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            for seed, opponent in zip(tqdm.trange(config.getint('sample', 'eval'), desc=f'{self.tag}_{self.cost}'), itertools.cycle(opponents)):
                with contextlib.closing(problem.evaluating(seed)):
                    controllers, ticks = problem.reset(self.kind, *opponent)
                    trajectory = loop.run_until_complete(asyncio.gather(
                        wuji.problem.mdp.rollout(controllers[0], agent, message=True),
                        *[wuji.problem.mdp._rollout(controller, agent) for controller, agent in zip(controllers[1:], opponent.values())],
                        *map(wuji.problem.mdp.ticking, ticks),
                    ))[0]
                    result = controllers[0].get_result()
                freq.append(wuji.problem.mdp.freq(init['kwargs']['outputs'], trajectory))
                for exp in trajectory:
                    for key, value in exp.items():
                        if torch.is_tensor(value):
                            exp[key] = value.detach().cpu()
                results.append(result)
                torch.save(dict(kind=self.kind, trajectory=trajectory, result=result), os.path.join(root, f'{seed}.pth'))
        wuji.file.tidy(os.path.dirname(root))
        freq = np.mean(freq, 0)
        result = problem.evaluate_reduce(results)
        wuji.record.mdp.image.Freq(self.cost, freq, names=init['action_name'] if 'action_name' in init else None, tag='/'.join([self.tag, 'freq']), **self.kwargs)(recorder)
        logging.info({key: value for key, value in result.items() if not key.startswith('_')})
        for key, value in result.items():
            if isinstance(value, (numbers.Integral, numbers.Real)) and not key.startswith('_'):
                recorder.writer.add_scalar('/'.join([self.tag, 'result', key]), value, self.cost)


def rollout(evaluator):
    class Evaluator(evaluator):
        def create_recorder(self):
            recorder = super().create_recorder()
            config = recorder.config_test
            recorder.register(config.get('record', 'rollout'), lambda: Rollout(self), config.getboolean('record', '_rollout'))
            return recorder
    return Evaluator
