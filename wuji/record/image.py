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
import mpl_toolkits.mplot3d


def to_image(fig):
    w, h = fig.canvas.get_width_height()
    image = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8).reshape([h, w, -1])
    plt.close(fig)
    return image[:, :, 1:]


class Scatter(object):
    def __init__(self, tag, step, points, *args, **kwargs):
        self.tag = tag
        self.step = step
        self.points = points
        self.args = args
        self.kwargs = kwargs

    def __call__(self, recorder):
        dim = len(self.points[0])
        assert dim > 1, dim
        fig = plt.figure()
        ax = mpl_toolkits.mplot3d.Axes3D(fig) if dim == 3 else fig.gca()
        ax.cla()
        if dim > 3:
            pass
        else:
            ax.scatter(*np.transpose(self.points), *self.args, **self.kwargs)
        fig.canvas.draw()
        image = to_image(fig)
        recorder.writer.add_image(self.tag, np.transpose(image, [2, 0, 1]), self.step)
