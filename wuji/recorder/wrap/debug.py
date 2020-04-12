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


def info(recorder):
    class Recorder(recorder):
        def create(self):
            try:
                logging.info('create recorder')
                return super().create()
            finally:
                logging.info('recorder created')

        def get(self, *args, **kwargs):
            record = super().get(*args, **kwargs)
            logging.info(f'get {record}')
            return record

        def tick(self):
            record = super().tick()
            logging.info(f'{record} recorded')
            return record

        def destroy(self):
            try:
                logging.info('destroy recorder')
                return super().destroy()
            finally:
                logging.info('recorder destroyed')
    return Recorder
