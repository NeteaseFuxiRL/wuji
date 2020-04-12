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

import torch.nn as nn

from ... import norm


def config(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for module in self.modules():
                self.init(module)

        def init(self, module):
            if isinstance(module, nn.Linear):
                module.weight = norm.uniform(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                module.weight = nn.init.kaiming_normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Embedding):
                module.weight = nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    return Module


def normal(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for module in self.modules():
                self.init(module)

        def init(self, module):
            if isinstance(module, nn.Linear):
                module.weight = nn.init.normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                module.weight = nn.init.normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Embedding):
                module.weight = nn.init.normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    return Module


def xavier_uniform(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for module in self.modules():
                self.init(module)

        def init(self, module):
            if isinstance(module, nn.Linear):
                module.weight = nn.init.xavier_uniform_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                module.weight = nn.init.xavier_uniform_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Embedding):
                module.weight = nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    return Module


def xavier_normal(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for module in self.modules():
                self.init(module)

        def init(self, module):
            if isinstance(module, nn.Linear):
                module.weight = nn.init.xavier_normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                module.weight = nn.init.xavier_normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Embedding):
                module.weight = nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    return Module


def kaiming_normal(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for module in self.modules():
                self.init(module)

        def init(self, module):
            if isinstance(module, nn.Linear):
                module.weight = nn.init.kaiming_normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                module.weight = nn.init.kaiming_normal_(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Embedding):
                module.weight = nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    return Module


def orthogonal(gain=1):
    def decorate(module):
        class Module(module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for module in self.modules():
                    self.init(module)

            def init(self, module):
                if isinstance(module, nn.Linear):
                    module.weight = nn.init.orthogonal_(module.weight, gain)
                    try:
                        module.bias.data.zero_()
                    except AttributeError:
                        pass
                elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                    module.weight = nn.init.orthogonal_(module.weight, gain)
                    try:
                        module.bias.data.zero_()
                    except AttributeError:
                        pass
                elif isinstance(module, nn.Embedding):
                    module.weight = nn.init.orthogonal_(module.weight, gain)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        return Module
    return decorate


def uniform(module):
    class Module(module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for module in self.modules():
                self.init(module)

        def init(self, module):
            if isinstance(module, nn.Linear):
                module.weight = norm.uniform(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                module.weight = norm.uniform(module.weight)
                try:
                    module.bias.data.zero_()
                except AttributeError:
                    pass
            elif isinstance(module, nn.Embedding):
                module.weight = norm.uniform(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    return Module
