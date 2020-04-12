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

import numpy as np
import matplotlib.pyplot as plt

from ..image import to_image


class Freq(object):
    def __init__(self, x, freq, tag='freq', names=None, **kwargs):
        self.x = x
        self.freq = freq
        self.tag = tag
        self.names = list(map(str, range(len(freq)))) if names is None else names
        self.kwargs = kwargs

    def __call__(self, recorder):
        fig = plt.figure(**self.kwargs)
        ax = fig.gca()
        ax.cla()
        x = np.arange(len(self.freq))
        self.plot(ax, x)
        ax.set_xticks(x)
        ax.set_xticklabels(self.names)
        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
        fig.canvas.draw()
        image = to_image(fig)
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.x)

    def plot(self, ax, x):
        assert len(self.freq.shape) == 2, self.freq.shape
        fail, success = self.freq.T
        total = success + fail
        rects = ax.bar(x, total, label='total')
        ax.bar(x, success, label='success')
        for _success, _total, rect in zip(success, total, rects):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), '%.1f/%.1f' % (_success, _total), ha='center', va='bottom')
