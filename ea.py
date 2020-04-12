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

import sys
import os
import argparse
import configparser
import logging
import logging.config
import contextlib
import shutil
import random
import subprocess
import functools
import tracemalloc
import gc
import traceback

import yaml
import torch
import ray
import tqdm
import humanfriendly
import filelock
import psutil

import wuji


def main():
    args = make_args()
    config = configparser.ConfigParser()
    for path in sum(args.config, []):
        wuji.config.load(config, path)
    for cmd in sum(args.modify, []):
        wuji.config.modify(config, cmd)
    with open(os.path.expanduser(os.path.expandvars(args.logging)), 'r') as f:
        logging.config.dictConfig(yaml.load(f))
    if args.log is None:
        args.log = wuji.config.digest(config)
    ea = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, config.get('ea', 'optimizer').split('\t')))
    root = os.path.expanduser(os.path.expandvars(config.get('model', 'root')))
    os.makedirs(root, exist_ok=True)
    with filelock.FileLock(root + '.lock', 0):
        if args.delete:
            logging.warning('delete model directory: ' + root)
            shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
    logging.info('cd ' + os.getcwd() + ' && ' + subprocess.list2cmdline([sys.executable] + sys.argv))
    logging.info('sys.path=' + ' '.join(sys.path))
    ray.init(**wuji.ray.init(config))
    timer = {key[5:]: (lambda: False) if value is None else wuji.counter.Time(humanfriendly.parse_timespan(value)) for key, value in vars(args).items() if key.startswith('time_')}
    if isinstance(timer['track'], wuji.counter.Time):
        tracemalloc.start()
    kwargs = {key: value for key, value in vars(args).items() if key not in {'config', 'modify'}}
    kwargs['root'] = root
    try:
        seed = config.getint('config', 'seed')
        wuji.random.seed(seed, prefix=f'seed={seed}: ')
        kwargs['seed'] = seed
    except configparser.NoOptionError:
        pass
    stopper = functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, config.get('ea', 'stopper').split('\t')))
    with contextlib.closing(ea(config, **kwargs)) as ea, contextlib.closing(stopper(ea)) as stopper, tqdm.tqdm(initial=ea.cost) as pbar, filelock.FileLock(root + '.lock', 0):
        root_log = ea.kwargs['root_log']
        os.makedirs(root_log, exist_ok=True)
        with open(root_log + '.ini', 'w') as f:
            config.write(f)
        logging.info(f'CUDA_VISIBLE_DEVICES= pushd "{os.path.dirname(root_log)}" && tensorboard --logdir {os.path.basename(root_log)}; popd')
        try:
            logging.info(', '.join([key + '=' + humanfriendly.format_size(functools.reduce(lambda x, wrap: wrap(x), map(wuji.parse.instance, encoding['agent']['eval'])).nbytes(random.choice(ea.population)['decision'][key]) if 'agent' in encoding else random.choice(ea.population)['decision'][key].nbytes) for key, encoding in ea.context['encoding'].items()]))
        except:
            traceback.print_exc()
        try:
            while True:
                outcome = ea(pbar=pbar)
                if timer['track']():
                    snapshot = tracemalloc.take_snapshot()
                    stats = snapshot.statistics('lineno')
                    for stat in stats[:10]:
                        print(stat)
                if timer['gc']():
                    gc.collect()
                    logging.warning('gc.collect')
                if stopper(outcome):
                    break
            logging.info('stopped')
        except KeyboardInterrupt:
            logging.warning('keyboard interrupted')
        ea.recorder(force=True)
        path = os.path.join(root, f'{ea.cost}.pth')
        logging.info(path)
        torch.save(ea.__getstate__(), path)
        try:
            wuji.file.tidy(root, config.getint('model', 'keep'))
        except configparser.NoOptionError:
            logging.warning(f'keep all models in {root}')


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', action='append', default=[[os.path.join(os.path.dirname(__file__), 'config.ini')]], help='config file')
    parser.add_argument('-m', '--modify', nargs='+', action='append', default=[], help='modify config')
    parser.add_argument('--logging', default=os.path.join(os.path.dirname(__file__), 'logging.yml'), help='logging config')
    parser.add_argument('-d', '--delete', action='store_true', help='delete model')
    parser.add_argument('-t', '--transfer', nargs='+', default=[], help='transfer blob')
    parser.add_argument('--log', help='the folder used to store log data')
    parser.add_argument('-p', '--parallel', type=int, default=psutil.cpu_count(logical=True))
    parser.add_argument('--time-track')
    parser.add_argument('--time-gc')
    parser.add_argument('--time-log')
    return parser.parse_args()


if __name__ == '__main__':
    main()
