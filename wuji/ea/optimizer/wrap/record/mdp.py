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
import numbers
import functools
import itertools
import copy
import configparser
import logging
import traceback

import numpy as np
import torch
import tqdm

import wuji.problem.mdp
import wuji.record.image
from wuji.rl.pth.wrap.opponent import make_agents
from wuji.evaluator.mdp.wrap.opponent import load


class Rollout(object):
    def __init__(self, optimizer, tag='rollout', **kwargs):
        self.optimizer = copy.deepcopy(optimizer.__getstate__())
        self.tag = tag
        self.kwargs = kwargs

    def __call__(self, recorder):
        config = recorder.config_test
        problem = recorder.problem
        population = self.optimizer['population']
        kind = self.kwargs.get('kind', 0)
        encoding = self.optimizer['context']['encoding']['blob']
        module = functools.reduce(lambda x, wrap: wrap(x), (wuji.parse.instance(m) if isinstance(m, str) else m for m in encoding['module']))
        model = module(config, **encoding['init'][kind]['kwargs'])
        agent = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, encoding['agent']['eval']))(model)
        try:
            kind, value = next(iter({int(key): value for key, value in config.items('opponent_file') if key.isdigit()}.items()))
            opponents = [make_agents(config, encoding, {kind: blob}) for blob in itertools.chain(*(load(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(os.path.expanduser(os.path.expandvars(value))) for filename in filenames))]
        except configparser.NoSectionError:
            opponents = [{}]
        for individual in tqdm.tqdm(population, desc=self.tag):
            model.set_blob(individual['decision']['blob'])
            cost, individual['result'] = wuji.problem.mdp.evaluate(problem, self.kwargs.get('kind', 0), agent, itertools.islice(itertools.cycle(opponents), config.getint('sample', 'eval')))
        root = os.path.join(os.path.expanduser(os.path.expandvars(config.get('model', 'root'))), 'eval')
        path = os.path.join(root, f'{self.optimizer["cost"]}.pth')
        logging.info(path)
        os.makedirs(root, exist_ok=True)
        torch.save(self.optimizer, path)
        try:
            wuji.file.tidy(root, config.getint('model', 'keep'))
        except configparser.NoOptionError:
            logging.warning(f'keep all models in {root}')
        point = eval('lambda result: ' + config.get('multi_objective', 'point'))
        points = np.array([point(individual['result']) for individual in population])
        color = eval('lambda result: ' + config.get('multi_objective', 'color'))
        color = np.array([color(individual['result']) for individual in population])
        recorder.put(wuji.record.image.Scatter('/'.join([self.tag, 'image']), self.optimizer['cost'], points, c=color))
        recorder.put(wuji.record.HistogramDict(self.optimizer['cost'], **{'/'.join([self.tag, 'result', key]): np.array([individual['result'][key] for individual in population]) for key, value in population[0]['result'].items() if not key.startswith('_') and isinstance(value, (numbers.Integral, numbers.Real))}))
        try:
            import pyotl
            ref = np.array(list(map(int, config.get('hypervolume', 'ref').split())))
            metric = pyotl.indicator.KMP_HV(-ref)(-points)
            recorder.put(wuji.record.Scalar(self.optimizer['cost'], **{'/'.join([self.tag, 'hypervolume']): metric}))
            logging.info(f'hypervolume={metric} (ref={ref})')
        except:
            traceback.print_exc()


def rollout(optimizer):
    class Optimizer(optimizer):
        def create_recorder(self):
            recorder = super().create_recorder()
            config = recorder.config_test
            recorder.register(config.get('record', 'rollout'), lambda: Rollout(self), config.getboolean('record', '_rollout'))
            return recorder
    return Optimizer
