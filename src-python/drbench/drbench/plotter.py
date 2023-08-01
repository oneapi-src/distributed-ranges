# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import re
from collections import namedtuple

import click
import pandas as pd
import seaborn as sns
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
        if fname.endswith('.rank000.json'):
            return True
        if re.match('.*\\.rank[0-9]{3}\\.json$', fname):
            return False
        return True

    @staticmethod
    def __import_file(fname: str, rows):
        with open(fname) as f:
            fdata = json.load(f)
            ctx = fdata['context']
            try:
                vsize = int(ctx['default_vector_size'])
                nprocs = int(ctx['ranks'])
                target = int(ctx['target'])
                model = int(ctx['model'])
                runtime = int(ctx['runtime'])
                device = int(ctx['device'])
            except KeyError:
                print(f'could not parse context of {fname}')
                raise
            benchs = fdata['benchmarks']
            for b in benchs:
                bname = b['name'].partition('/')[0]
                rtime = b['real_time']
                bw = b['bytes_per_second']
                rows.append(
                    {
                        'target': target,
                        'model': model,
                        'runtime': runtime,
                        'device': device,
                        'vsize': vsize,
                        'benchmark': bname,
                        'nprocs': nprocs,
                        'rtime': rtime,
                        'bw': bw,
                    }
                )

    # db is created which looks something like this:
    #           mode  vsize          benchmark  nprocs      rtime            bw
    # 0   MHP_NOSYCL  20000   Stream_Copy       1   0.234987  1.361779e+11
    # 1   MHP_NOSYCL  20000  Stream_Scale       1   0.240879  1.328468e+11
    # 2   MHP_NOSYCL  20000    Stream_Add       1   0.329298  1.457645e+11
    # ..         ...    ...           ...     ...        ...           ...
    # 62     MHP_GPU  40000    Stream_Add       4  21.716973  4.420506e+09
    # 63     MHP_GPU  40000  Stream_Triad       4  21.714421  4.421025e+09
    def __init__(self, plotting_config: PlottingConfig):
        rows = []
        for fname in glob.glob('dr-bench*.json'):
            click.echo(f'found file {fname}')
            Plotter.__import_file(fname, rows)

        self.db = pd.DataFrame(rows)

        # helper structures that can be used to define plots
        self.vec_sizes = self.db['vsize'].unique()
        self.vec_sizes.sort()
        self.max_vec_size = self.vec_sizes[-1]
        self.db_maxvec = self.db.loc[(self.db['vsize'] == self.max_vec_size)]

        self.nprocs = self.db['nprocs'].unique()
        self.nprocs.sort()

        self.modes = self.db['mode'].unique()

    @staticmethod
    def __make_plot(fname, data, **kwargs):
        plot = sns.relplot(data=data, kind='line', **kwargs)
        plot.savefig(f'{fname}.png')

    def __stream_bandwidth_plots(self):
        Plotter.__make_plot(
            'stream_bw',
            self.db_maxvec.loc[self.db['benchmark'].str.startswith('Stream_')],
            x='nprocs',
            y='bw',
            col='mode',
            hue='benchmark',
        )

    def __stream_strong_scaling_plots(self):
        db = self.db_maxvec.loc[
            self.db['benchmark'].str.startswith('Stream_')
        ].copy()

        ref_stream = sorted(db['benchmark'].unique())[0]
        ref_mode = sorted(db['mode'].unique())[0]
        ref_nproc = sorted(db['nprocs'].unique())[0]
        # take value of reference stream/mode/nproc - can it be easier taken?
        scale_factor = (
            db.loc[
                (db['mode'] == ref_mode)
                & (db['benchmark'] == ref_stream)
                & (db['nprocs'] == ref_nproc)
            ]
            .squeeze()
            .at['bw']
        )

        click.echo(
            f'stream strong scalling scalled by {ref_stream} {ref_mode}'
            f' nproc:{ref_nproc} eq {scale_factor}'
        )
        db['bw'] /= scale_factor

        Plotter.__make_plot(
            'stream_strong_scaling',
            db,
            x='nprocs',
            y='bw',
            col='benchmark',
            hue='mode',
        )

    def create_plots(self):
        sns.set_theme(style="ticks")

        self.__stream_bandwidth_plots()
        self.__stream_strong_scaling_plots()
