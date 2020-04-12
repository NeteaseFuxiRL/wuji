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
import logging

import numpy as np
import mpl_toolkits.mplot3d

import wuji.problem


class Visualizer(wuji.Visualizer):
    def __init__(self, config, boundary, background):
        self.config = config
        self.boundary = boundary
        self.background = background
        super().__init__(config.getfloat('visualizer', 'interval'))
        section = '_'.join(os.path.splitext(__file__)[0].split(os.path.sep)[-2:])
        self.size = tuple(map(int, config.get(section, 'size').split()))

    def plot(self, fig, *args, **kwargs):
        dim = len(self.boundary)
        if dim == 1:
            self.plot2(fig, *args, **kwargs)
        elif dim == 2:
            self.plot3(fig, *args, **kwargs)

    def plot2(self, fig, *args, **kwargs):
        ax = fig.gca()
        ax.cla()
        if self.background is not None:
            self.background(ax)
        try:
            marker = kwargs['marker']
        except KeyError:
            marker = ['.']
        for population, marker in zip(args, itertools.cycle(marker)):
            ax.scatter(*zip(*[np.concatenate([individual['decision'], [individual['result']['fitness']]]) for individual in population]), marker=marker)
        ax.set_xlim(self.boundary[0])

    def plot3(self, fig, *args, **kwargs):
        ax = mpl_toolkits.mplot3d.Axes3D(fig)
        for name in ['x', 'y', 'z']:
            getattr(ax, f'set_{name}label')(name)
        ax.cla()
        if self.background is not None:
            self.background(ax)
        try:
            marker = kwargs['marker']
        except KeyError:
            marker = ['.']
        for population, marker in zip(args, itertools.cycle(marker)):
            ax.scatter(*zip(*[np.concatenate([individual['decision'], [individual['result']['fitness']]]) for individual in population]), marker=marker)
        ax.set_xlim(self.boundary[0])
        ax.set_ylim(self.boundary[1])
        if hasattr(self, 'ax'):
            ax.view_init(self.ax.elev, self.ax.azim)
        self.ax = ax


def fitness(optimizer):
    class Optimizer(optimizer):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            assert len(self.context['encoding']) == 1, list(self.context['encoding'].keys())
            self.boundary = next(iter(self.context['encoding'].values()))['boundary']
            self.problem = wuji.problem.create(config)
            self.visualizer = self.create_visualizer()

        def close(self):
            self.visualizer.close()

        def create_visualizer(self):
            if 'DISPLAY' in os.environ:
                try:
                    background = self.problem.background
                except AttributeError:
                    background = lambda *args, **kwargs: None
                visualizer = Visualizer(self.config, self.boundary, background)
                visualizer.start()
            else:
                logging.warning('display not available, use a fake visualizer')
                visualizer = lambda *args, **kwargs: None
                visualizer.close = lambda: None
            return visualizer

        def __call__(self, *args, **kwargs):
            try:
                return super().__call__(*args, **kwargs)
            finally:
                self.visualizer(
                    list(map(self.visualize_point, self.offspring)),
                    list(map(self.visualize_point, self.population)),
                    [self.visualize_point(self.elite)],
                    marker=['.', '.', '*'],
                )

        def visualize_point(self, individual):
            return dict(decision=next(iter(individual['decision'].values())), result=individual['result'])
    return Optimizer
