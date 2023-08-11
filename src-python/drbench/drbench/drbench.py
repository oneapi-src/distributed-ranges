#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import click
from drbench import common, plotter, runner


class SuiteConfig:
    def __init__(self):
        self.prefix = None
        self.filter = None
        self.reps = None
        self.dry_run = None
        self.mhp_bench = None
        self.shp_bench = None
        self.weak_scaling = False
        self.nodes = 1
        self.ranks = 1
        self.ranks_per_node = 0
        self.target = None
        self.vec_size = None


option_nodes = click.option(
    "--nodes",
    type=int,
    default=1,
    help="Number of Nodess",
)

option_ranks_per_node = click.option(
    "--ranks-per-node", type=int, help="ranks per node"
)

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

option_weak_scaling = click.option(
    "--weak-scaling",
    is_flag=True,
    default=False,
    help="Scales the vector size by the number of ranks",
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
    click.echo(f"Targets: {options.target}")
    click.echo(f"Ranks: {options.ranks}")
    r = runner.Runner(
        runner.AnalysisConfig(
            options.prefix,
            "\\|".join(options.filter),
            options.reps,
            options.dry_run,
            options.mhp_bench,
            options.shp_bench,
            options.weak_scaling,
            options.ranks_per_node,
        )
    )
    for t in options.target:
        for s in options.vec_size:
            for n in options.ranks:
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
    help="Run with 1 ... N ranks",
)
@option_ranks_per_node
@click.option("--reps", default=50, type=int, help="Number of reps")
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
@option_weak_scaling
def run(
    prefix,
    target,
    vec_size,
    ranks,
    rank_range,
    ranks_per_node,
    reps,
    filter,
    mhp_bench,
    shp_bench,
    dry_run,
    clean,
    weak_scaling,
):
    assert target
    assert vec_size
    assert ranks

    options = SuiteConfig()
    if clean and not dry_run:
        do_clean(prefix)

    options.prefix = prefix
    options.target = target
    options.vec_size = vec_size
    options.ranks_per_node = ranks_per_node
    options.ranks = ranks
    if rank_range:
        options.ranks = list(range(1, rank_range + 1))
    options.reps = reps
    options.filter = filter
    options.mhp_bench = mhp_bench
    options.shp_bench = shp_bench
    options.weak_scaling = weak_scaling
    options.dry_run = dry_run

    do_run(options)


@cli.command()
@option_prefix
@option_mhp_bench
@option_shp_bench
@option_dry_run
@option_clean
@option_weak_scaling
@click.option(
    "--vec-size",
    type=int,
    default=1000000000,
    help="Size of a vector",
)
@click.option(
    "--reps",
    type=int,
    default=50,
    help="Number of repetitions",
)
@option_nodes
@click.option(
    "--gpus",
    type=int,
    default=0,
    help="Number of GPUs per node",
)
@click.option(
    "--no-p2p",
    "p2p",
    is_flag=True,
    default=True,
    help="Do not run benchmarks that require p2p",
)
@click.option(
    "--cores-per-socket",
    type=int,
    default=0,
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
    nodes,
    gpus,
    p2p,
    cores_per_socket,
    weak_scaling,
):
    # Base options
    base = SuiteConfig()
    base.prefix = prefix
    base.mhp_bench = mhp_bench
    base.shp_bench = shp_bench
    base.dry_run = dry_run
    base.vec_size = [vec_size]
    base.reps = reps
    base.weak_scaling = weak_scaling
    base.nodes = nodes

    # Run a range of ranks
    def suite_run_rank_list(ranks_per_node, ranks, filters, targets):
        options = base
        options.ranks_per_node = ranks_per_node
        options.ranks = ranks
        options.filter = filters
        options.target = targets
        do_run(options)

    # Run a range of ranks
    def suite_run_rank_range(ranks_per_node, filters, targets):
        suite_run_rank_list(
            ranks_per_node,
            list(range(1, ranks_per_node * base.nodes + 1)),
            filters,
            targets,
        )

    dr_nop2p = [
        "^Stream_",
        "^Black_Scholes",
    ]

    dr_p2p = [
        "^Inclusive_Scan_DR",
        "^Reduce_DR",
    ]

    dr_filters = dr_nop2p + dr_p2p

    # if the platform does not support p2p, limit gpus to 1
    if p2p:
        p2p_gpus = gpus
    else:
        p2p_gpus = min(gpus, 1)

    if clean and not dry_run:
        do_clean(prefix)

    #
    # GPU devices
    #
    if gpus > 0:
        # if benchmark does not need p2p run xhp on all gpus
        suite_run_rank_range(gpus, dr_nop2p, ["shp_sycl_gpu", "mhp_sycl_gpu"])
        # if benchmark needs p2p run on mhp on all gpus
        suite_run_rank_range(gpus, dr_p2p, ["mhp_sycl_gpu"])
        # DPL is 1 device
        suite_run_rank_list(1, [1], [".*_DPL"], ["shp_sycl_gpu"])
    if p2p_gpus > 0:
        # if benchmark needs p2p run on shp on 1 gpu
        suite_run_rank_range(1, dr_p2p, ["shp_sycl_gpu"])

    #
    # CPU devices
    #
    if cores_per_socket > 0:
        # treat each socket as a device, assume 2 sockets
        suite_run_rank_range(2, dr_filters, ["mhp_sycl_cpu", "shp_sycl_cpu"])
        # DPL is 1 device
        suite_run_rank_range(2, [".*_DPL"], ["shp_sycl_cpu"])
    if cores_per_socket > 0:
        # 1 and 2 sockets for direct cpu
        suite_run_rank_list(
            2,
            [cores_per_socket, 2 * cores_per_socket],
            dr_filters,
            ["mhp_direct_cpu"],
        )


if __name__ == "__main__":
    assert False  # not to be used this way, but by dr-bench executable
