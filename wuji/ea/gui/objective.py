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

import logging

import numpy as np
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


class Selector(object):
    def __init__(self, config, population, picker=3):
        self.config = config
        self.population = population
        point = eval('lambda result: ' + config.get('multi_objective', 'point'))
        self.points = np.array([point(individual['result']) for individual in population])
        color = eval('lambda result: ' + config.get('multi_objective', 'color'))
        color = np.array([color(individual['result']) for individual in population])
        self.fig = plt.figure()
        cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(np.min(color), np.max(color)), cmap=matplotlib.cm.jet)
        dim = len(self.points[0])
        if dim == 3:
            ax = mpl_toolkits.mplot3d.Axes3D(self.fig)
            for name in ['x', 'y', 'z']:
                getattr(ax, f'set_{name}label')(name)
        else:
            ax = self.fig.gca()
        if dim > 3:
            x = list(range(dim))
            self.artist = [ax.plot(x, point, color=c, picker=picker)[0] for point, c in zip(self.points, cmap.to_rgba(color))]
            ax.set_xticks(x, minor=False)
        else:
            self.artist = [ax.scatter(*point, color=c, picker=picker) for point, c in zip(self.points, cmap.to_rgba(color))]
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.artist_target = []

    def close(self):
        return plt.close(self.fig)

    def on_pick(self, event):
        for artist in self.artist_target:
            artist.remove()
        self.fig.canvas.draw()
        index = self.artist.index(event.artist)
        self.individual = self.population[index]
        logging.info(f'individual{index}: {self.individual["result"]}')

    def on_click(self, event):
        if event.button == 3:  # right click
            for artist in self.artist_target:
                artist.remove()
            self.artist_target = []
            target = np.array([event.xdata, event.ydata])
            ax = self.fig.gca()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            self.artist_target.append(ax.plot(*target, '*')[0])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            self.fig.canvas.draw()
            self.fig.canvas.set_window_title(', '.join(map(str, target)))
