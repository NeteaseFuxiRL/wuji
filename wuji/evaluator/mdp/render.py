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
import argparse
import configparser
import logging
import logging.config
import contextlib
import types
import random
import asyncio

import yaml
import numpy as np
import torch

import wuji
import wuji.agent
import wuji.problem.mdp
from wuji.problem.mdp.wrapper import Render


def main():
    args = make_args()
    config = configparser.ConfigParser()
    for path in sum(args.config, []):
        wuji.config.load(config, path)
    if args.test:
        config = wuji.config.test(config)
    for cmd in sum(args.modify, []):
        wuji.config.modify(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    if args.log is None:
        args.log = wuji.config.digest(config)
    if args.seed >= 0:
        logging.warning(f'use random seed {args.seed}')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    root = os.path.expanduser(os.path.expandvars(config.get('model', 'root')))
    loop = asyncio.get_event_loop()
    with contextlib.closing(wuji.problem.create(config)) as problem:
        path, x = wuji.file.load(root)
        logging.info('load ' + path)
        data = torch.load(path, map_location=lambda storage, loc: storage)
        with torch.no_grad(), contextlib.closing(wuji.agent.load(config, data, args.kind)) as agent:
            for seed in range(np.iinfo(np.int).max):
                with contextlib.closing(problem.evaluating(seed) if not args.training else types.SimpleNamespace(close=lambda: None)):
                    (controller,), ticks = problem.reset(args.kind)
                    controller = Render(controller, fps=args.fps)
                    loop.run_until_complete(asyncio.gather(
                        wuji.problem.mdp._rollout(controller, agent, message=True),
                        *map(wuji.problem.mdp.ticking, ticks),
                    ))
                    result = controller.get_result()
                logging.info(result)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.sep.join(__file__.split(os.sep)[:-4] + ['config.ini'])]], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[], help='modify config')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('--logging', default='logging.yml', help='logging config')
    parser.add_argument('--log', help='the folder used to store log data')
    parser.add_argument('--kind', type=int, default=0)
    parser.add_argument('-f', '--fps', type=int, default=0)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
