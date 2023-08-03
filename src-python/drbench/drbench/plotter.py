# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import re

import click
import pandas as pd
import seaborn as sns


class Plotter:
    bandwidth_title = 'Memory Bandwidth (TBps)'

    @staticmethod
    def __name_target(bname, target, device):
        print(f"In: {bname} {target} {device}")
        names = bname.split('_')
        last = names[-1]
        if last in ["DPL", "Std", "Serial"]:
            bname = "_".join(names[0:-1])
            target = f"{last}_{device}"
        elif last == "DR":
            bname = "_".join(names[0:-1])
        print(f"out: {bname} {target} {device}")
        return bname, target

    @staticmethod
    def __import_file(fname: str, rows):
        with open(fname) as f:
            fdata = json.load(f)
            ctx = fdata['context']
            try:
                vsize = int(ctx['default_vector_size'])
                ranks = int(ctx['ranks'])
                target = ctx['target']
                model = ctx['model']
                runtime = ctx['runtime']
                device = ctx['device']
            except KeyError:
                print(f'could not parse context of {fname}')
                raise
            benchs = fdata['benchmarks']
            cores_per_socket = int(
                re.search(
                    r'Core\(s\) per socket:\s*(\d+)', ctx['lscpu']
                ).group(1)
            )
            for b in benchs:
                bname = b['name'].partition('/')[0]
                bname, btarget = Plotter.__name_target(bname, target, device)
                rtime = b['real_time']
                bw = b['bytes_per_second']
                rows.append(
                    {
                        'Target': btarget,
                        'model': model,
                        'runtime': runtime,
                        'device': device,
                        'vsize': vsize,
                        'Benchmark': bname,
                        'Ranks': ranks,
                        'GPU Tiles': ranks,
                        'CPU Sockets': ranks / cores_per_socket
                        if runtime == 'DIRECT' and device == 'CPU'
                        else ranks,
                        'rtime': rtime,
                        Plotter.bandwidth_title: bw / 1e12,
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
    def __init__(self, prefix):
        rows = []
        for fname in glob.glob(f'{prefix}-*.json'):
            click.echo(f'found file {fname}')
            Plotter.__import_file(fname, rows)

        self.db = pd.DataFrame(rows)

        # helper structures that can be used to define plots
        self.vec_sizes = self.db['vsize'].unique()
        self.vec_sizes.sort()
        self.max_vec_size = self.vec_sizes[-1]
        self.db_maxvec = self.db.loc[(self.db['vsize'] == self.max_vec_size)]

        self.ranks = self.db['Ranks'].unique()
        self.ranks.sort()

    @staticmethod
    def __make_plot(fname, data, **kwargs):
        plot = sns.relplot(data=data, kind='line', marker='d', **kwargs)
        plot.savefig(f'{fname}.png', dpi=200, bbox_inches='tight')

    def __stream_strong_scaling_plots(self):
        db = self.db_maxvec.loc[
            self.db['Benchmark'].str.startswith('Stream_')
        ].copy()

        db_gpu = db.loc[db['device'] == 'GPU']

        Plotter.__make_plot(
            'stream_strong_scaling_gpu',
            db_gpu,
            x='GPU Tiles',
            y=Plotter.bandwidth_title,
            col='Benchmark',
            style='Target',
        )

        db_cpu = db.loc[db['device'] == 'CPU']

        Plotter.__make_plot(
            'stream_strong_scaling_cpu',
            db_cpu,
            x='CPU Sockets',
            y=Plotter.bandwidth_title,
            col='Benchmark',
            style='Target',
        )

    def __algorithm_plots(self):
        db = self.db_maxvec.loc[
            self.db['Benchmark'].str.startswith('Inclusive_Scan')
        ].copy()

        db_gpu = db.loc[db['device'] == 'GPU']

        Plotter.__make_plot(
            'algorithms_gpu',
            db_gpu,
            x='GPU Tiles',
            y=Plotter.bandwidth_title,
            col='Benchmark',
            style='Target',
        )

        db_cpu = db.loc[db['device'] == 'CPU']

        Plotter.__make_plot(
            'algorithms_cpu',
            db_cpu,
            x='CPU Sockets',
            y=Plotter.bandwidth_title,
            col='Benchmark',
            style='Target',
        )

    def create_plots(self):
        sns.set_theme(style="ticks")

        self.__stream_strong_scaling_plots()
        self.__algorithm_plots()
