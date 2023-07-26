# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections import namedtuple

import click
from drbench import common

# only common_config for now, add plotting options here if needed
PlottingConfig = namedtuple(
    'PlottingConfig',
    'common_config',
)


class Plotter:
    @staticmethod
    def __is_our_file(fname: str, analysis_id: str):
        files_prefix = common.analysis_file_prefix(analysis_id)
        if not fname.startswith(files_prefix):
            return False
        if fname.startswith(files_prefix + '.rank000'):
            return True
        if fname.startswith(files_prefix + '.rank'):
            return False
        return True

    def __init__(self, plotting_config: PlottingConfig):
        for fname in os.listdir('.'):
            if Plotter.__is_our_file(
                fname, plotting_config.common_config.analysis_id
            ):
                click.echo(f'found file {fname}')

    def create_plots(self):
        pass
