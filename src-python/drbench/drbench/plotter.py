# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import re
from collections import namedtuple

import click
import pandas as pd
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

    @staticmethod
    def __import_file(fname: str, rows):
        mode = re.search(r'AnalysisMode\.([A-Z_]+)\.', fname).group(1)
        with open(fname) as f:
            fdata = json.load(f)
            ctx = fdata['context']
            vsize = int(ctx['default_vector_size'])
            nprocs = int(ctx['nprocs'])
            benchs = fdata['benchmarks']
            for b in benchs:
                bname = b['name'].partition('/')[0]
                rtime = b['real_time']
                bw = b['bytes_per_second']
                rows.append(
                    {
                        'mode': mode,
                        'vsize': vsize,
                        'test': bname,
                        'nprocs': nprocs,
                        'rtime': rtime,
                        'bw': bw,
                    }
                )

    def __init__(self, plotting_config: PlottingConfig):
        rows = []
        for fname in os.listdir('.'):
            if Plotter.__is_our_file(
                fname, plotting_config.common_config.analysis_id
            ):
                click.echo(f'found file {fname}')
                Plotter.__import_file(fname, rows)
        self.db = pd.DataFrame(rows)

    def create_plots(self):
        print(self.db)
