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
@click.option('--nprocs', default=1, type=int, help='Number of processes')
@click.option('--sycl-gpu', is_flag=True, help='Use sycl')
@click.option('--sycl-cpu', is_flag=True, help='Use sycl on cpu device')
@click.option(
    '--bench', default='./mhp-bench', type=str, help='Benchmark program'
)
@click.pass_context
def mhp(ctx, nprocs, sycl_gpu, sycl_cpu, bench):
    if sycl_gpu:
        # mhp-bench will spread GPUs over ranks automatically, so no
        # pinning ONEAPI_DEVICE_SELECTOR is workaround for devcloud
        # multi-card not working
        env = 'ONEAPI_DEVICE_SELECTOR=level_zero:0'
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
        f'{env} mpirun -n {nprocs} '
        f'{bench} {sycl_args} {ctx.obj["COMMON_ARGS"]} '
        f'--benchmark_out={ctx.obj["OUTPUT"]}'
    )
    execute(command, ctx)


if __name__ == '__main__':
    cli(obj={})
