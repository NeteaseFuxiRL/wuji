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
import logging

import numpy as np
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import humanfriendly

import wuji.record
import wuji.ea.pareto


def extract(population, point, color):
    assert len(np.array(point(population[0]['result'])).shape) == 1, point(population[0]['result'])
    assert isinstance(color(population[0]['result']), (numbers.Integral, numbers.Real)), color(population[0]['result'])
    return [dict(result=individual['result'], point=point(individual['result']), color=color(individual['result'])) for individual in population]


def plot(ax, population):
    color = [individual['color'] for individual in population]
    cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
    dim = len(population[0]['point'])
    if dim > 3:
        x = list(range(dim))
        for individual in population:
            point = individual['point']
            ax.plot(x, point, color=cmap.to_rgba(individual['color']), picker=1)
        ax.set_xticks(x, minor=False)
    else:
        color = cmap.to_rgba([individual['color'] for individual in population])
        points = [individual['point'] for individual in population]
        ax.scatter(*zip(*points), c=color, picker=1)
    return cmap


def point(selection):
    class Selection(selection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, 'plot_point')
            self.plot_point = eval('lambda result: ' + self.config.get('multi_objective', 'point'))
    return Selection


def color(selection):
    class Selection(selection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, 'plot_color')
            self.plot_color = eval('lambda result: ' + self.config.get('multi_objective', 'color'))
    return Selection


class Record(object):
    def __init__(self, tag, x, population, **kwargs):
        self.tag = tag
        self.x = x
        self.population = population
        self.kwargs = kwargs

    def __call__(self, recorder):
        image = self.draw()
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.x)

    def plot(self):
        figsize = self.kwargs['figsize'] if 'figsize' in self.kwargs else None
        dpi = self.kwargs['dpi'] if 'dpi' in self.kwargs else None
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if len(self.population[0]['point']) == 3:
            ax = mpl_toolkits.mplot3d.Axes3D(fig)
            for name in ['x', 'y', 'z']:
                getattr(ax, f'set_{name}label')(name)
        else:
            ax = fig.gca()
        ax.cla()
        plot(ax, self.population)
        return fig

    def draw(self):
        fig = self.plot()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8).reshape([h, w, -1])
        plt.close(fig)
        return image[:, :, 1:]


def record(selection):
    class Selection(selection):
        def __call__(self, *args, **kwargs):
            self.selected = super().__call__(*args, **kwargs)
            return self.selected

        def record(self, optimizer):
            super().record(optimizer)
            optimizer.recorder.register(self.config.get('record', 'plot'), lambda: Record('multi_objective/selected', optimizer.cost, extract(self.selected, self.plot_point, self.plot_color)))
    return Selection


class Visualizer(wuji.Visualizer):
    def __init__(self, config):
        super().__init__(config.getfloat('visualizer', 'interval'))
        self.config = config

    def create(self):
        fig = super().create()
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        return fig

    def plot(self, fig, population):
        dim = len(population[0]['point'])
        if dim == 3:
            ax = mpl_toolkits.mplot3d.Axes3D(fig)
            for name in ['x', 'y', 'z']:
                getattr(ax, f'set_{name}label')(name)
        else:
            ax = fig.gca()
        ax.cla()
        cmap = plot(ax, population)
        if dim == 3 and hasattr(self, 'ax'):
            ax.view_init(self.ax.elev, self.ax.azim)
        self.ax = ax
        fig.canvas.set_window_title(f'({cmap.norm.vmin}, {cmap.norm.vmax})')
        self._population = population

    def on_pick(self, event):
        for i in event.ind:
            individual = self._population[i]
            logging.info(f'individual{i}: {individual["result"]}')


def visualize(selection):
    class Selection(selection):
        def __init__(self, config):
            super().__init__(config)
            if 'DISPLAY' in os.environ:
                self.visualizer = Visualizer(self.config)
                self.visualizer.start()
            else:
                logging.warning('display not available, use a fake visualizer')
                self.visualizer = lambda *args, **kwargs: None
                self.visualizer.close = lambda: None

        def close(self):
            self.visualizer.close()

        def __call__(self, population, size):
            try:
                return super().__call__(population, size)
            finally:
                self.visualizer(extract(self.selected, self.plot_point, self.plot_color))
    return Selection
