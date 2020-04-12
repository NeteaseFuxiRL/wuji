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
import configparser

import scipy.stats

NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def ttest(selection):
    class Selection(selection):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            try:
                self.dominate = getattr(self, 'dominate_' + config.get(NAME, 'ttest_dominate'))
            except (configparser.NoOptionError, AttributeError):
                pass
            self.pvalue = config.getfloat('ttest', 'pvalue')

        def dominate(self, individual1, individual2):
            return super().dominate(individual1, individual2) and scipy.stats.ttest_ind(individual1['sample']['fitness'], individual2['sample']['fitness'])[1] < self.pvalue

        def dominate_any(self, individual1, individual2):
            assert len(individual1['sample']['objective'][0]) == len(individual2['sample']['objective'][0]), (len(individual1['sample']['objective'][0]), len(individual2['sample']['objective'][0]))
            return super().dominate(individual1, individual2) and any(scipy.stats.ttest_ind([objective[i] for objective in individual1['sample']['objective']], [objective[i] for objective in individual2['sample']['objective']])[1] < self.pvalue for i in range(len(individual1['sample']['objective'][0])))

        def dominate_all(self, individual1, individual2):
            assert len(individual1['sample']['objective'][0]) == len(individual2['sample']['objective'][0]), (len(individual1['sample']['objective'][0]), len(individual2['sample']['objective'][0]))
            return super().dominate(individual1, individual2) and all(scipy.stats.ttest_ind([objective[i] for objective in individual1['sample']['objective']], [objective[i] for objective in individual2['sample']['objective']])[1] < self.pvalue for i in range(len(individual1['sample']['objective'][0])))
    return Selection
