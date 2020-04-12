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

from ...mating import Tournament
from wuji.ea.pareto import *


class NSGA_II(Tournament):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dominate = eval('lambda individual1, individual2: ' + self.config.get('non_dominated', 'dominate'))
        self.density = eval('lambda individual: ' + self.config.get('nsga_ii', 'density'))

    def compete(self, item1, item2):
        index1, individual1 = item1
        index2, individual2 = item2
        if self.dominate(individual1, individual2):
            return item1
        elif self.dominate(individual2, individual1):
            return item2
        else:
            try:
                if self.density(individual1) > self.density(individual2):
                    return item1
                else:
                    return item2
            except KeyError:
                return self.random.choice([item1, item2])
