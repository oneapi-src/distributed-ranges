#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import logging
import os
import sys

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
        self.different_devices = False
        self.ranks = 1
        self.ranks_per_node = None
        self.target = None
        self.vec_size = None
        self.device_memory = False


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

option_different_devices = click.option(
    "--different-devices",
    is_flag=True,
    default=False,
    help="Ensures there are not multiple ranks on one SYCL device",
)


# common arguments
@click.group()
def cli():
    logging.basicConfig(
        stream=sys.stdout,
        format=(
            "%(levelname)s %(asctime)s" "[%(filename)s:%(lineno)d] %(message)s"
        ),
        datefmt="%Y-%m-%d_%H:%M:%S",
        level=logging.INFO,
    )


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
    r = runner.Runner(
        runner.AnalysisConfig(
            options.prefix,
            "\\|".join(options.filter),
            options.reps,
            options.dry_run,
            options.mhp_bench,
            options.shp_bench,
            options.weak_scaling,
            options.different_devices,
            options.ranks_per_node,
            options.device_memory,
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
@click.option(
    "--node-range",
    type=int,
    help="Run with 1 ... N nodes",
)
@click.option("--ranks-per-node", type=int, help="Ranks per node")
@click.option("--reps", default=50, type=int, help="Number of reps")
@click.option(
    "-f",
    "--filter",
    type=str,
    multiple=True,
    default=["Stream_"],
    help="A filter used for a benchmark",
)
@click.option("--device-memory", is_flag=True, help="Use device memory")
@option_mhp_bench
@option_shp_bench
@option_dry_run
@option_clean
@option_weak_scaling
@option_different_devices
def run(
    prefix,
    target,
    vec_size,
    ranks,
    rank_range,
    node_range,
    ranks_per_node,
    reps,
    filter,
    device_memory,
    mhp_bench,
    shp_bench,
    dry_run,
    clean,
    weak_scaling,
    different_devices,
):
    assert target
    assert vec_size
    assert ranks

    options = SuiteConfig()
    if clean and not dry_run:
        do_clean(prefix)

    if node_range and not ranks_per_node:
        click.get_current_context().fail(
            "--node-range requires --ranks-per-node"
        )

    options.device_memory = device_memory
    options.prefix = prefix
    options.target = target
    options.vec_size = vec_size
    options.ranks_per_node = ranks_per_node
    options.ranks = ranks
    if rank_range:
        options.ranks = list(range(1, rank_range + 1))
    if node_range:
        options.ranks = list(
            range(
                ranks_per_node, node_range * ranks_per_node + 1, ranks_per_node
            )
        )
    options.reps = reps
    options.filter = filter
    options.mhp_bench = mhp_bench
    options.shp_bench = shp_bench
    options.weak_scaling = weak_scaling
    options.different_devices = different_devices
    options.dry_run = dry_run

    do_run(options)


@cli.command()
@option_prefix
@option_mhp_bench
@option_shp_bench
@option_dry_run
@option_clean
@option_weak_scaling
@option_different_devices
@click.option(
    "--vec-size",
    type=int,
    default=2000000000,
    help="Size of a vector",
)
@click.option(
    "--reps",
    type=int,
    default=50,
    help="Number of repetitions",
)
@click.option(
    "--nodes",
    type=int,
    help="Number of nodes",
)
@click.option(
    "--gpus",
    type=int,
    default=0,
    help="Number of GPUs per node",
)
@click.option(
    "--sockets",
    type=int,
    default=0,
    help="Number of CPU sockets per node",
)
@click.option(
    "--cores-per-socket",
    type=int,
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
    sockets,
    cores_per_socket,
    weak_scaling,
    different_devices,
):
    # Run a list of ranks
    def run_rank_list(
        base,
        ranks,
        filter,
        targets,
        weak_scaling_filter=[],
        device_memory=False,
    ):
        options = base
        options.ranks = ranks
        options.filter = filter
        options.target = targets
        options.weak_scaling = False
        options.device_memory = device_memory
        do_run(options)

        weak = list(set(filter) & set(weak_scaling_filter))
        if len(weak) > 0:
            options.filter = weak
            options.weak_scaling = True
            do_run(options)

    # Run a range of ranks on a single node
    def run_rank_range(base, ranks, filters, targets, weak_scaling=False):
        run_rank_list(
            base,
            list(range(1, ranks + 1)),
            filters,
            targets,
            weak_scaling,
        )

    # Run sequence 1, 2, 4, 8, 12 based on total ranks
    def run_rank_sparse(
        base,
        ranks,
        filters,
        targets,
        weak_scaling_filter=[],
        device_memory=False,
    ):
        run_rank_list(
            base,
            list(filter(lambda r: r <= ranks, [1, 2, 4, 8, 12])),
            filters,
            targets,
            weak_scaling_filter=weak_scaling_filter,
            device_memory=device_memory,
        )

    # Run a range of nodes
    def run_node_range(base, ranks_per_node, filters, targets):
        options = base
        options.ranks_per_node = ranks_per_node
        run_rank_list(
            options,
            list(
                range(
                    ranks_per_node, ranks_per_node * nodes + 1, ranks_per_node
                )
            ),
            filters,
            targets,
        )

    def single_node(base):
        #
        # GPU devices
        #
        if gpus > 0:
            # benchmarks
            run_rank_sparse(
                base,
                gpus,
                xhp_filter + mhp_filter,
                ["mhp_sycl_gpu"],
                weak_scaling_filter,
            )
            run_rank_sparse(
                base,
                gpus,
                device_memory_filter,
                ["mhp_sycl_gpu"],
                device_memory=True,
            )
            run_rank_sparse(
                base,
                gpus,
                xhp_filter + shp_filter,
                ["shp_sycl_gpu"],
                weak_scaling_filter,
                device_memory=True,
            )

            # reference
            run_rank_sparse(base, 1, mhp_reference_filter, ["mhp_sycl_gpu"])
            run_rank_sparse(
                base,
                1,
                sycl_reference_filter + shp_reference_filter,
                ["shp_sycl_gpu"],
            )

        #
        # CPU devices
        #
        if sockets > 0:
            # SYCL on CPU, a socket (Affinity domain) is a device
            run_rank_range(
                base,
                sockets,
                xhp_filter + mhp_filter,
                ["mhp_sycl_cpu"],
            )
            run_rank_range(
                base,
                sockets,
                xhp_filter + shp_filter,
                ["shp_sycl_cpu"],
            )
            # Run reference benchmarks on 1 device, use shp_sycl_cpu to
            # get sycl env vars
            run_rank_range(base, 1, sycl_reference_filter, ["shp_sycl_cpu"])
            run_rank_range(base, 1, mhp_reference_filter, ["mhp_sycl_cpu"])
            # 1 and 2 sockets for direct cpu
            run_rank_list(
                base,
                list(
                    range(
                        cores_per_socket,
                        sockets * cores_per_socket + 1,
                        cores_per_socket,
                    )
                ),
                mhp_filter + xhp_filter,
                ["mhp_direct_cpu"],
            )

    def multi_node(base):
        #
        # GPU devices
        #
        if gpus > 0:
            run_node_range(
                base, gpus, xhp_filter + mhp_filter, ["mhp_sycl_gpu"]
            )

        #
        # CPU devices
        #
        if sockets > 0:
            run_node_range(base, sockets, xhp_filter, ["mhp_sycl_cpu"])
            run_node_range(
                base,
                sockets * cores_per_socket,
                xhp_filter,
                ["mhp_direct_cpu"],
            )

    # benchmark filters
    xhp_filter = [
        "BlackScholes_DR",
        "DotProduct_DR",
        # does not work
        "Exclusive_Scan_DR"
        "Inclusive_Scan_DR",
        "Reduce_DR",
        "Stream_Triad",
    ]
    weak_scaling_filter = [
        "Exclusive_Scan_DR",
        "Inclusive_Scan_DR",
        "Reduce_DR",
        ".*Sort_DR",
        "WaveEquation_DR",
        "Gemm_Reference",
    ]
    mhp_filter = ["Stencil2D_DR", "WaveEquation_DR"]
    device_memory_filter = ["FFT3D_DR", "WaveEquation_DR"]
    shp_filter = ["FFT3D_DR", ".*Sort_DR", "Gemm_DR"]
    # reference benchmarks that do not use shp or mhp
    sycl_reference_filter = [
        "BlackScholes_Reference",
        "DotProduct_Reference",
        "Inclusive_Scan_Reference",
        "Reduce_Reference",
    ]
    mhp_reference_filter = [
        "Stencil2D_Reference",
    ]
    shp_reference_filter = [
        ".*Sort_Reference",
        "Gemm_Reference",
    ]

    if sockets and not cores_per_socket:
        click.get_current_context().fail(
            "--sockets requires --cores-per-socket"
        )

    # Base options
    base = SuiteConfig()
    base.prefix = prefix
    base.mhp_bench = mhp_bench
    base.shp_bench = shp_bench
    base.dry_run = dry_run
    base.vec_size = [vec_size]
    base.reps = reps
    base.weak_scaling = weak_scaling
    base.different_devices = different_devices

    logging.info(
        f"starting suite, weak_scaling:{weak_scaling}, "
        f"different_devices:{different_devices}"
    )

    if clean and not dry_run:
        do_clean(prefix)

    if nodes:
        multi_node(base)
    else:
        single_node(base)


if __name__ == "__main__":
    assert False  # not to be used this way, but by dr-bench executable
