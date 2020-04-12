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
import shutil
import hashlib
import operator
import logging


def tidy(root, keep=5):
    prefixes = {name.split('.')[0] for name in os.listdir(root)}
    prefixes = [int(prefix) for prefix in prefixes if prefix.isdigit()]
    if len(prefixes) > keep:
        prefixes = sorted(prefixes)
        remove = prefixes[:-keep]
        for prefix in map(str, remove):
            for name in os.listdir(root):
                if name.startswith(prefix):
                    path = os.path.join(root, name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)


def load(root, step=None, ext='.pth', logger=logging.info):
    """
    Load the latest checkpoint in a model directory.
    :author 申瑞珉 (Ruimin Shen)
    :real root: The directory to store the model checkpoint files.
    :real step: If a integer value is given, the corresponding checkpoint will be loaded. Otherwise, the latest checkpoint (with the largest step value) will be loaded.
    :real ext: The extension of the model file.
    :return:
    """
    if step is None:
        steps = [(int(n), n) for n, e in map(os.path.splitext, os.listdir(root)) if n.isdigit() and e == ext]
        step, name = max(steps, key=operator.itemgetter(0))
    else:
        name = str(step)
    prefix = os.path.join(root, name)
    if logger is not None:
        logger(f'load {prefix}.*')
    path = prefix + ext
    assert os.path.exists(path), path
    return path, step


def group_filename(comp, concat='-'):
    if len(comp) == 1:
        return comp[0]
    prefix = os.path.commonprefix(comp)
    if prefix:
        comp = [s[len(prefix):] for s in comp]
        return f'{prefix}[{concat.join([comp[0], group_filename(comp[1:], concat)])}]'
    else:
        return concat.join([comp[0], group_filename(comp[1:], concat)])


def short_filename(name, max=255, join='|'):
    if len(name) > max:
        digest = hashlib.md5(name.encode()).hexdigest()
        return name[:max - len(join) - len(digest)] + join + digest
    else:
        return name
