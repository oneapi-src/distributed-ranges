#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import subprocess

import click


def execute(command: str, ctx):
    click.echo(command)
    if not ctx.obj['DRY_RUN']:
        subprocess.run(command, shell=True)


# common arguments
@click.group()
@click.option('--vec-size', default=1000000, type=int, help='Size of a vector')
@click.option('--reps', default=100, type=int, help='Number of reps')
@click.option(
    '--benchmark-filter',
    default='Stream_',
    type=str,
    help='A filter used for a benchmark',
)
@click.option(
    '--dry-run', is_flag=True, help='Emits commands but does not execute'
)
@click.option(
    '--timestamp', is_flag=True, help='Output is dr-bench-{timestamp}.json'
)
@click.option(
    '--output', default='dr-bench.json', type=str, help='Output json file'
)
@click.pass_context
def cli(ctx, vec_size, reps, benchmark_filter, dry_run, timestamp, output):
    ctx.ensure_object(dict)
    ctx.obj['COMMON_ARGS'] = (
        f'--vector-size {str(vec_size)} --reps {str(reps)} '
        f'--benchmark_out_format=json '
        f'--benchmark_filter={benchmark_filter}'
    )
    ctx.obj['DRY_RUN'] = dry_run
    now = datetime.datetime.now().isoformat(timespec="minutes")
    ctx.obj['OUTPUT'] = f'dr-bench-{now}.json' if timestamp else output


# mhp subcommand
@cli.command()
@click.option(
    '--bench', default='./mhp-bench', type=str, help='Benchmark program'
)
@click.option('--fork', is_flag=True, help='Use -launcher=fork with mpi')
@click.option('--nprocs', default=1, type=int, help='Number of processes')
@click.option('--sycl-cpu', is_flag=True, help='Use sycl on cpu device')
@click.option('--sycl-gpu', is_flag=True, help='Use sycl')
@click.pass_context
def mhp(ctx, bench, fork, nprocs, sycl_cpu, sycl_gpu):
    if sycl_gpu:
        # mhp-bench will spread GPUs over ranks automatically, so no
        # pinning is needed
        env = 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu'
        sycl_args = '--sycl'
    elif sycl_cpu:
        env = 'ONEAPI_DEVICE_SELECTOR=opencl:cpu'
        sycl_args = '--sycl'
    else:
        env = (
            'I_MPI_PIN_DOMAIN=core I_MPI_PIN_ORDER=compact I_MPI_PIN_CELL=unit'
        )
        sycl_args = ''

    command = (
        f'{env} mpirun {"-launcher=fork" if fork else ""} -n {nprocs} '
        f'{bench} {sycl_args} {ctx.obj["COMMON_ARGS"]} '
        f'--benchmark_out=mhp_{nprocs}_{ctx.obj["OUTPUT"]}'
    )
    execute(command, ctx)

    plot_command = 'python Plotter.py --mhp'
    execute(plot_command, ctx)


# shp subcommand
@cli.command()
@click.option(
    '--bench', default='./shp-bench', type=str, help='Benchmark program'
)
@click.option('--d', default=1, type=int, help='Number of devices')
@click.pass_context
def shp(ctx, bench, d):
    env = 'KMP_AFFINITY=compact'

    command = (
        f'{env} {bench} -d {d} {ctx.obj["COMMON_ARGS"]} '
        f'--benchmark_out=shp_{d}_{ctx.obj["OUTPUT"]}'
    )
    execute(command, ctx)

    plot_command = 'python Plotter.py --shp'
    execute(plot_command, ctx)


# plot (both shp and mhp) subcommand without running benchmarks
@cli.command
@click.pass_context
def plot_only(ctx):
    plot_command = 'python Plotter.py --mhp --shp'
    execute(plot_command, ctx)


if __name__ == '__main__':
    cli(obj={})
