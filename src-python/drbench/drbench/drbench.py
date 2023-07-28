#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime

import click
from drbench import common, plotter, runner


# common arguments
@click.group()
@click.option(
    '--analysis_id',
    type=str,
    default='',
    help='id to use in output files, use time based if missing',
)
@click.pass_context
def cli(ctx, analysis_id: str):
    ctx.obj = common.Config()
    if analysis_id:
        ctx.obj.analysis_id = analysis_id
    else:
        ctx.obj.analysis_id = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H_%M_%S'
        )


def __plot_impl(ctx):
    p = plotter.Plotter(plotter.PlottingConfig(ctx.obj))
    p.create_plots()


@cli.command()
@click.pass_context
def plot(ctx):
    __plot_impl(ctx)


Choice = click.Choice(['mhp_cpu', 'mhp_gpu', 'mhp_nosycl', 'shp'])


def choice_to_mode(c):
    if c == 'mhp_cpu':
        return runner.AnalysisMode.MHP_CPU
    if c == 'mhp_gpu':
        return runner.AnalysisMode.MHP_GPU
    if c == 'mhp_nosycl':
        return runner.AnalysisMode.MHP_NOSYCL
    if c == 'shp':
        return runner.AnalysisMode.SHP
    assert False


@cli.command()
@click.option(
    '--no-plot',
    'plot',
    default=True,
    is_flag=True,
    help="don't create plots, just json files",
)
@click.option(
    '-m',
    '--mode',
    type=Choice,
    multiple=True,
    default=['mhp_cpu'],
    help='modes of benchmarking to run',
)
@click.option(
    '-s',
    '--vec-size',
    type=int,
    multiple=True,
    default=[1000000],
    help='Size of a vector',
)
@click.option(
    '-n',
    '--nprocs',
    type=int,
    multiple=True,
    default=[1],
    help='Number of processes',
)
@click.option('--fork', is_flag=True, help='Use -launcher=fork with mpi')
@click.option('-r', '--reps', default=100, type=int, help='Number of reps')
@click.option(
    '-f',
    '--benchmark-filter',
    default='Stream_',
    type=str,
    help='A filter used for a benchmark',
)
@click.option(
    '--mhp-bench',
    default='mhp/mhp-bench',
    type=str,
    help='MHP benchmark program',
)
@click.option(
    '--shp-bench',
    default='shp/shp-bench',
    type=str,
    help='SHP benchmark program',
)
@click.option(
    '-d', '--dry-run', is_flag=True, help='Emits commands but does not execute'
)
@click.pass_context
def analyse(
    ctx,
    plot,
    mode,
    vec_size,
    nprocs,
    fork,
    reps,
    benchmark_filter,
    mhp_bench,
    shp_bench,
    dry_run,
):
    assert mode
    assert vec_size
    assert nprocs

    r = runner.Runner(
        runner.AnalysisConfig(
            ctx.obj,
            benchmark_filter,
            fork,
            reps,
            dry_run,
            mhp_bench,
            shp_bench,
        )
    )
    for m in mode:
        for s in vec_size:
            for n in nprocs:
                r.run_one_analysis(
                    runner.AnalysisCase(choice_to_mode(m), s, n)
                )

    if plot:
        __plot_impl(ctx)


if __name__ == '__main__':
    assert False  # not to be used this way, but by dr-bench executable
