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
import operator

import numpy as np
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import wuji

NAME = os.path.basename(os.path.dirname(os.path.splitext(__file__)[0]))


def plot(ax, layers, **kwargs):
    population = list(itertools.chain(*layers))
    _color = kwargs['color']
    color = [individual['result'][_color] for individual in population]
    cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
    dim = len(population[0]['result']['objective'])
    if dim > 3:
        x = list(range(dim))
        for index, (layer, marker) in enumerate(zip(layers, itertools.cycle(kwargs['marker_line']))):
            for individual in layer:
                objective = individual['result']['objective']
                if isinstance(objective[0], tuple):
                    objective = tuple(map(operator.itemgetter(0), objective))
                artist, = ax.plot(x, objective, color=cmap.to_rgba(individual['result'][_color]), picker=1)
                assert not hasattr(artist, 'index')
                artist.index = index
        ax.set_xticks(x, minor=False)
    else:
        for index, (layer, marker) in enumerate(zip(layers, itertools.cycle(kwargs['marker_point']))):
            color = cmap.to_rgba([individual['result'][_color] for individual in layer])
            front = [individual['result']['objective'] for individual in layer]
            if isinstance(front[0][0], tuple):
                front = [tuple(map(operator.itemgetter(0), objective)) for objective in front]
            artist = ax.scatter(*zip(*front), c=color, marker=marker, picker=1)
            assert not hasattr(artist, 'index')
            artist.index = index
    return cmap


class Visualizer(wuji.Visualizer):
    def __init__(self, config):
        super().__init__(config.getfloat('visualizer', 'interval'))
        self.config = config
        self.color = config.get(NAME, 'color')
        self.marker_point = config.get(NAME, 'marker_point').split()
        self.marker_line = config.get(NAME, 'marker_line').split()

    def create(self):
        fig = super().create()
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        return fig

    def plot(self, fig, layers):
        dim = len(layers[0][0]['result']['objective'])
        if dim == 3:
            ax = mpl_toolkits.mplot3d.Axes3D(fig)
            for name in ['x', 'y', 'z']:
                getattr(ax, f'set_{name}label')(name)
        else:
            ax = fig.gca()
        ax.cla()
        cmap = plot(ax, layers, color=self.color, marker_point=self.marker_point, marker_line=self.marker_line)
        if dim == 3 and hasattr(self, 'ax'):
            ax.view_init(self.ax.elev, self.ax.azim)
        self.ax = ax
        fig.canvas.set_window_title(f'{self.color}: ({cmap.norm.vmin}, {cmap.norm.vmax})')
        self._layers = layers

    def on_pick(self, event):
        layer = self._layers[event.artist.index]
        for i in event.ind:
            individual = layer[i]
            print(individual['sample'], individual['result'])
