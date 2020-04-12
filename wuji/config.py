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
import io
import re
import hashlib
import configparser
import logging

prog = re.compile(r'\{([a-zA-Z0-9_]+)\/([a-zA-Z0-9_]+)\}')


def text(config):
    with io.StringIO() as buffer:
        config.write(buffer)
        return buffer.getvalue()


def load(config, path):
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    assert os.path.exists(path), path
    with open(path, 'r') as f:
        s = f.read()
    s = s.replace('${PATH}', path)
    s = s.replace('${PREFIX}', os.path.splitext(path)[0])
    s = s.replace('${DIRNAME}', os.path.dirname(path))
    basename = os.path.basename(path)
    s = s.replace('${BASENAME}', basename)
    s = s.replace('${FILENAME}', os.path.splitext(basename)[0])
    try:
        s = prog.sub(lambda m: config.get(m.group(1), m.group(2)), s)
        config.read_string(s)
    except:
        logging.fatal(path)
        raise


def modify(config, cmd):
    try:
        var, value = cmd.split('=', 1)
        section, option = var.split('/', 1)
    except ValueError:
        logging.fatal(cmd)
        raise
    if value:
        try:
            value = prog.sub(lambda m: config.get(m.group(1), m.group(2)), value)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logging.fatal(value)
            raise
        try:
            config.set(section, option, value)
        except configparser.NoSectionError:
            config.add_section(section)
            config.set(section, option, value)
    else:
        try:
            if option:
                config.remove_option(section, option)
            else:
                config.remove_section(section)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass


def test(config):
    try:
        for value in filter(None, config.get('config', 'test').split('\t')):
            path = os.path.expandvars(os.path.expanduser(value))
            if os.path.exists(path):
                load(config, path)
            else:
                modify(config, value)
    except (configparser.NoSectionError, configparser.NoOptionError):
        pass
    return config


def digest(config):
    return hashlib.md5(text(config).encode()).hexdigest()
