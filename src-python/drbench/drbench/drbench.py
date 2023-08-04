#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import click
from drbench import common, plotter, runner

option_prefix = click.option(
    "--prefix",
    type=str,
    default="dr-bench",
    help="Prefix for files",
)

option_mhp_bench = click.option(
    "--mhp-bench",
    default="mhp/mhp-bench",
    type=str,
    help="MHP benchmark program",
)

option_shp_bench = click.option(
    "--shp-bench",
    default="shp/shp-bench",
    type=str,
    help="SHP benchmark program",
)

option_dry_run = click.option(
    "-d", "--dry-run", is_flag=True, help="Emits commands but does not execute"
)

option_clean = click.option(
    "-c", "--clean", is_flag=True, help="Delete all json files with the prefix"
)


# common arguments
@click.group()
def cli():
    pass


@cli.command()
@option_prefix
def plot(prefix):
    p = plotter.Plotter(prefix)
    p.create_plots()


def do_clean(prefix):
    for f in glob.glob(f"{prefix}-*.json"):
        os.remove(f)


@cli.command()
@option_prefix
def clean(prefix):
    do_clean(prefix)


Choice = click.Choice(common.targets.keys())


def choice_to_target(c):
    return common.targets[c]


def do_run(options):
    if options['rank_range'] > 0:
        options['ranks'] = list(range(1, options['rank_range'] + 1))

    click.echo(f"Targets: {options['target']}")
    click.echo(f"Ranks: {options['ranks']}")
    r = runner.Runner(
        runner.AnalysisConfig(
            options['prefix'],
            "\\|".join(options['filter']),
            options['reps'],
            options['dry_run'],
            options['mhp_bench'],
            options['shp_bench'],
        )
    )
    for t in options['target']:
        for s in options['vec_size']:
            for n in options['ranks']:
                r.run_one_analysis(
                    runner.AnalysisCase(choice_to_target(t), s, n)
                )


@cli.command()
@option_prefix
@click.option(
    "--target",
    type=Choice,
    multiple=True,
    default=["mhp_direct_cpu"],
    help="Target to execute benchmark",
)
@click.option(
    "--vec-size",
    type=int,
    multiple=True,
    default=[1000000],
    help="Size of a vector",
)
@click.option(
    "--ranks",
    type=int,
    multiple=True,
    default=[1],
    help="Number of ranks",
)
@click.option(
    "--rank-range",
    type=int,
    default=0,
    help="Run with 1 ... N ranks",
)
@click.option("--reps", default=100, type=int, help="Number of reps")
@click.option(
    "-f",
    "--filter",
    type=str,
    multiple=True,
    default=["Stream_"],
    help="A filter used for a benchmark",
)
@option_mhp_bench
@option_shp_bench
@option_dry_run
@option_clean
def run(
    prefix,
    target,
    vec_size,
    ranks,
    rank_range,
    reps,
    filter,
    mhp_bench,
    shp_bench,
    dry_run,
    clean,
):
    assert target
    assert vec_size
    assert ranks

    options = {}
    if clean and not dry_run:
        do_clean(prefix)

    do_run(
        options
        | {
            'prefix': prefix,
            'target': target,
            'vec_size': vec_size,
            'ranks': ranks,
            'rank_range': rank_range,
            'reps': reps,
            'filter': filter,
            'mhp_bench': mhp_bench,
            'shp_bench': shp_bench,
            'dry_run': dry_run,
        }
    )


@cli.command()
@option_prefix
@option_mhp_bench
@option_shp_bench
@option_dry_run
@option_clean
@click.option(
    "--vec-size",
    type=int,
    default=1000000000,
    help="Size of a vector",
)
@click.option(
    "--reps",
    type=int,
    default=100,
    help="Number of repetitions",
)
@click.option(
    "--gpus",
    type=int,
    default=1,
    help="Number of GPUs",
)
@click.option(
    "--p2p-gpus",
    type=int,
    default=-1,
    help="Number of GPUs for p2p",
)
@click.option(
    "--cores-per-socket",
    type=int,
    default=2,
    help="Number of cores per CPU socket",
)
def suite(
    prefix,
    mhp_bench,
    shp_bench,
    dry_run,
    clean,
    vec_size,
    reps,
    gpus,
    p2p_gpus,
    cores_per_socket,
):
    # Base options
    base = {
        'prefix': prefix,
        'mhp_bench': mhp_bench,
        'shp_bench': shp_bench,
        'dry_run': dry_run,
        'vec_size': [vec_size],
        'reps': reps,
    }

    def suite_run(rank_range, filters, targets):
        do_run(
            base
            | {'rank_range': rank_range, 'filter': filters, 'target': targets}
        )

    def suite_run_ranks(ranks, filters, targets):
        do_run(
            base
            | {
                'ranks': ranks,
                'rank_range': 0,
                'filter': filters,
                'target': targets,
            }
        )

    dr_nop2p = [
        '^Stream_',
        '^Black_Scholes',
    ]

    dr_p2p = [
        '^Inclusive_Scan_DR',
        '^Reduce_DR',
    ]

    dr_filters = dr_nop2p + dr_p2p

    if p2p_gpus == -1:
        p2p_gpus = gpus

    if clean and not dry_run:
        do_clean(prefix)

    if gpus > 0:
        # DPL is 1 device
        suite_run(1, ['.*_DPL'], ['shp_sycl_gpu'])
        suite_run(gpus, dr_nop2p, ['shp_sycl_gpu', 'mhp_sycl_gpu'])
        suite_run(gpus, dr_p2p, ['mhp_sycl_gpu'])

    if p2p_gpus > 0:
        suite_run(p2p_gpus, dr_p2p, ['shp_sycl_gpu'])

    if cores_per_socket > 0:
        suite_run(1, ['.*_DPL'], ['shp_sycl_cpu'])
        suite_run(2, dr_filters, ['mhp_sycl_cpu', 'shp_sycl_cpu'])

    # 1 and 2 sockets for direct cpu
    if cores_per_socket > 0:
        suite_run_ranks(
            [cores_per_socket, 2 * cores_per_socket],
            dr_filters,
            ['mhp_direct_cpu'],
        )


if __name__ == "__main__":
    assert False  # not to be used this way, but by dr-bench executable
