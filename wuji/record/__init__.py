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
import collections
import json
import configparser
import inspect
import logging
import traceback

import numpy as np
import torch
import pandas as pd
import scipy.spatial.distance
import matplotlib.pyplot as plt
import tinydb
import xlsxwriter
import inflection

import wuji


def save_xlsx(df, path, results, worksheet='worksheet'):
    with xlsxwriter.Workbook(path, {'strings_to_urls': False, 'nan_inf_to_errors': True}) as workbook:
        worksheet = workbook.add_worksheet(worksheet)
        for j, key in enumerate(df):
            worksheet.write(0, j, key)
            try:
                result = results[key]
            except (KeyError, AttributeError):
                result = None
            if hasattr(result, 'add_format'):
                fmt = result.add_format(workbook, worksheet)
            else:
                fmt = None
            for i, value in enumerate(df[key]):
                worksheet.write(1 + i, j, value, fmt)
            if hasattr(result, 'conditional_format'):
                result.conditional_format(workbook, worksheet, i, j)
        worksheet.autofilter(0, 0, i, len(df.columns) - 1)
        worksheet.freeze_panes(1, 0)


class Save(object):
    def __init__(self, evaluator):
        root = os.path.expanduser(os.path.expandvars(evaluator.config.get('model', 'root')))
        path = os.path.join(root, f'{evaluator.cost}.pth')
        logging.info(path)
        os.makedirs(root, exist_ok=True)
        torch.save(evaluator.__getstate__(), path)
        try:
            wuji.file.tidy(root, evaluator.config.getint('model', 'keep'))
        except configparser.NoOptionError:
            logging.warning(f'keep all models in {root}')


class Metric(object):
    def __init__(self, metric, *args, **kwargs):
        self.metric = metric
        self.args = args
        self.kwargs = kwargs

    def __call__(self, recorder):
        name = inflection.underscore(type(self).__name__)
        path = wuji.get_metric_db(recorder.config_test)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results = self.save_db(recorder, path)
        with open(path, 'r') as f:
            df = pd.read_json(json.dumps(json.load(f)['_default']), orient='index', convert_dates=False)
        df = df[sorted(df)]
        try:
            df = df.sort_values(recorder.config_test.get(name, 'sort'))
        except configparser.NoOptionError:
            pass
        save_xlsx(df, os.path.splitext(path)[0] + '.xlsx', collections.OrderedDict(results))

    def save_db(self, recorder, path):
        results = []
        with tinydb.TinyDB(path) as db:
            item = []
            for metric in self.metric:
                metric = metric(recorder, *self.args, **self.kwargs)
                for name, member in inspect.getmembers(metric, predicate=inspect.ismethod):
                    if not name.startswith('_'):
                        try:
                            result = member()
                            value = result()
                            if isinstance(value, np.floating):
                                value = float(value)
                            item.append((name, value))
                            results.append((name, result))
                        except:
                            logging.warning(name)
                            traceback.print_exc()
                try:
                    if hasattr(metric, '__call__'):
                        metric(recorder)
                except:
                    traceback.print_exc()
            db.insert(dict(item))
        return results


class Scalar(object):
    def __init__(self, x, **kwargs):
        self.x = x
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, value in self.kwargs.items():
            recorder.writer.add_scalar(key, value, self.x)


class Scalars(object):
    def __init__(self, x, tag, **kwargs):
        self.x = x
        self.tag = tag
        self.kwargs = kwargs

    def __call__(self, recorder):
        recorder.writer.add_scalars(self.tag, self.kwargs, self.x)


class Vector(object):
    def __init__(self, x, **kwargs):
        self.x = x
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, vector in self.kwargs.items():
            for i, value in enumerate(vector):
                recorder.writer.add_scalar(f'{key}{i}', value, self.x)


class Flat(object):
    def __init__(self, x, **kwargs):
        self.x = x
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, points in self.kwargs.items():
            if points.size:
                matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points))
                recorder.writer.add_scalar('/'.join([key, 'dist_mean']), np.mean(matrix), self.x)
                recorder.writer.add_scalar('/'.join([key, 'dist_max']), np.max(matrix), self.x)


class HistogramDict(object):
    def __init__(self, x, **kwargs):
        self.x = x
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, var in self.kwargs.items():
            recorder.writer.add_histogram(key, var, self.x)


class HistogramList(object):
    def __init__(self, x, **kwargs):
        self.x = x
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, l in self.kwargs.items():
            for i, a in enumerate(l):
                if a.size:
                    recorder.writer.add_histogram(f'{key}{i}', a, self.x)


class Embedding(object):
    def __init__(self, x, tag, **kwargs):
        self.x = x
        self.tag = tag
        self.kwargs = kwargs

    def __call__(self, recorder):
        try:
            cmap = self.kwargs['cmap']
        except KeyError:
            cmap = plt.cm.jet
        try:
            repeat = self.kwargs['repeat']
        except KeyError:
            repeat = 3
        data, label = (self.kwargs[key] for key in 'data, label'.split(', '))
        if 'fitness' in self.kwargs:
            fitness = self.kwargs['fitness']
            a, b = min(fitness), max(fitness)
            r = b - a
            if r > 0:
                colors = [cmap(int((s - a) / r * cmap.N)) for s in fitness]
                images = torch.from_numpy(np.reshape([color[:3] for color in colors], [-1, 3, 1, 1]).repeat(repeat, 2).repeat(repeat, 3))
                images = (images * 255).float()
            else:
                images = None
        else:
            images = None
        recorder.writer.add_embedding(data, label, label_img=images, global_step=self.x, tag=self.tag)


class Text(object):
    def __init__(self, x, **kwargs):
        self.x = x
        self.kwargs = kwargs

    def __call__(self, recorder):
        for key, value in self.kwargs.items():
            recorder.writer.add_text(key, value, self.x)


class Distribution(object):
    def __init__(self, x, population, **kwargs):
        self.x = x
        self.population = population
        self.kwargs = kwargs

    def __call__(self, recorder):
        name = inflection.underscore(type(self).__name__)
        stamp_index = {individual['stamp']: i for i, individual in enumerate(self.population)}
        for tag, ids in self.kwargs.items():
            for i, id in enumerate(ids):
                individual = self.population[stamp_index[id]]
                if 'tags' in individual:
                    individual['tags'][tag] = i
                else:
                    individual['tags'] = {tag: i}
        root = os.path.join(recorder.root_model, name)
        os.makedirs(root, exist_ok=True)
        torch.save(self.population, os.path.join(root, f'{self.x}.pth'))
