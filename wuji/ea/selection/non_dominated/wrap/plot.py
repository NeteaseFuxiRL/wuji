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
import itertools
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

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def extract_layers(layers, point, color):
    assert len(np.array(point(layers[0][0]['result'])).shape) == 1, point(layers[0][0]['result'])
    assert isinstance(color(layers[0][0]['result']), (numbers.Integral, numbers.Real)), color(layers[0][0]['result'])
    return [[dict(point=point(individual['result']), color=color(individual['result'])) for individual in layer] for layer in layers]


def plot(ax, layers, **kwargs):
    flat = list(itertools.chain(*layers))
    color = [data['color'] for data in flat]
    cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
    dim = len(flat[0]['point'])
    if dim > 3:
        x = list(range(dim))
        for index, (layer, marker) in enumerate(zip(layers, itertools.cycle(kwargs['marker_line']))):
            for data in layer:
                point = data['point']
                artist, = ax.plot(x, point, color=cmap.to_rgba(data['color']), picker=1)
                assert not hasattr(artist, 'index')
                artist.index = index
        ax.set_xticks(x, minor=False)
    else:
        for index, (layer, marker) in enumerate(zip(layers, itertools.cycle(kwargs['marker_point']))):
            color = cmap.to_rgba([data['color'] for data in layer])
            points = [data['point'] for data in layer]
            artist = ax.scatter(*zip(*points), c=color, marker=marker, picker=1)
            assert not hasattr(artist, 'index')
            artist.index = index
    return cmap


def create(selection):
    class Selection(selection):
        def __init__(self, config, *args, **kwargs):
            self.point = eval('lambda result: ' + config.get('multi_objective', 'point'))
            self.color = eval('lambda result: ' + config.get('multi_objective', 'color'))
            self.marker_point = config.get(NAME, 'marker_point').split()
            self.marker_line = config.get(NAME, 'marker_line').split()
            super().__init__(config, *args, **kwargs)
    return Selection


class Record(object):
    def __init__(self, tag, x, layers, **kwargs):
        self.tag = tag
        self.x = x
        self.layers = layers
        self.kwargs = kwargs

    def __call__(self, recorder):
        image = self.draw()
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.x)

    def plot(self):
        figsize = self.kwargs['figsize'] if 'figsize' in self.kwargs else None
        dpi = self.kwargs['dpi'] if 'dpi' in self.kwargs else None
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if len(self.layers[0][0]['point']) == 3:
            ax = mpl_toolkits.mplot3d.Axes3D(fig)
            for name in ['x', 'y', 'z']:
                getattr(ax, f'set_{name}label')(name)
        else:
            ax = fig.gca()
        ax.cla()
        plot(ax, self.layers, **self.kwargs)
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
        def record(self, optimizer):
            super().record(optimizer)
            optimizer.recorder.register(self.config.get('record', 'plot'), lambda: Record('non_dominated/selected', optimizer.cost, extract_layers(self.layers[:-1] + [self.critical], self.point, self.color), marker_point=self.marker_point, marker_line=self.marker_line))
    return Selection


class Visualizer(wuji.Visualizer):
    def __init__(self, config, **kwargs):
        super().__init__(config.getfloat('visualizer', 'interval'))
        self.config = config
        self._kwargs = kwargs

    def create(self):
        fig = super().create()
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        return fig

    def plot(self, fig, layers):
        dim = len(layers[0][0]['point'])
        if dim == 3:
            ax = mpl_toolkits.mplot3d.Axes3D(fig)
            for name in ['x', 'y', 'z']:
                getattr(ax, f'set_{name}label')(name)
        else:
            ax = fig.gca()
        ax.cla()
        cmap = plot(ax, layers, **self._kwargs)
        if dim == 3 and hasattr(self, 'ax'):
            ax.view_init(self.ax.elev, self.ax.azim)
        self.ax = ax
        fig.canvas.set_window_title(f'({cmap.norm.vmin}, {cmap.norm.vmax})')
        self._layers = layers

    def on_pick(self, event):
        layer = self._layers[event.artist.index]
        for i in event.ind:
            individual = layer[i]
            print(individual['sample'], individual['result'])


def visualize(selection):
    class Selection(selection):
        def __init__(self, config):
            super().__init__(config)
            if 'DISPLAY' in os.environ:
                self.visualizer = Visualizer(self.config, marker_point=self.marker_point, marker_line=self.marker_line)
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
                self.visualizer(extract_layers(self.layers[:-1] + [self.critical], self.point, self.color))
    return Selection
