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
    '--mode',
    type=Choice,
    multiple=True,
    default=['mhp_cpu'],
    help='modes of benchmarking to run',
)
@click.option(
    '--vec-size',
    type=int,
    multiple=True,
    default=[1000000],
    help='Size of a vector',
)
@click.option(
    '--nprocs',
    type=int,
    multiple=True,
    default=[1],
    help='Number of processes',
)
@click.option('--fork', is_flag=True, help='Use -launcher=fork with mpi')
@click.option('--reps', default=100, type=int, help='Number of reps')
@click.option(
    '--benchmark-filter',
    default='Stream_',
    type=str,
    help='A filter used for a benchmark',
)
@click.option(
    '--mhp-bench',
    default='./mhp-bench',
    type=str,
    help='MHP benchmark program',
)
@click.option(
    '--shp-bench',
    default='./shp-bench',
    type=str,
    help='SHP benchmark program',
)
@click.option(
    '--dry-run', is_flag=True, help='Emits commands but does not execute'
)
@click.pass_context
def analyse(
    ctx,
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


@cli.command()
@click.pass_context
def plot(ctx):
    plotter.do_nothing()  # TODO: to be implemented


if __name__ == '__main__':
    assert False  # not to be used this way, but by dr-bench executable
