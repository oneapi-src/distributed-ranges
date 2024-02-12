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
        self.retries = None
        self.dry_run = None
        self.mhp_bench = None
        self.shp_bench = None
        self.weak_scaling = False
        self.different_devices = False
        self.ranks = 1
        self.target = None
        self.vec_size = None
        self.device_memory = False


option_prefix = click.option(
    "--prefix",
    type=str,
    default="dr-bench",
    help="Prefix for files",
)

option_ppn = click.option(
    "--ppn",
    type=int,
    default=2,
    help="Number of processes per node",
)

option_reps = click.option(
    "--reps", default=50, type=int, help="Number of reps"
)

option_retries = click.option(
    "--retries",
    type=int,
    default=3,
    help="count of retries for failed analysis",
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
    # github log annotation
    click.echo("::group::dr-bench plot")
    p = plotter.Plotter(prefix)
    p.create_plots()
    click.echo("::endgoup::")


def do_clean(prefix):
    for f in glob.glob(f"{prefix}-*.json"):
        os.remove(f)


@cli.command()
@option_prefix
def clean(prefix):
    # github log annotation
    click.echo("::group::dr-bench clean")
    do_clean(prefix)
    click.echo("::endgroup::")


Choice = click.Choice(common.targets.keys())


def choice_to_target(c):
    return common.targets[c]


def do_run(options):
    r = runner.Runner(
        runner.AnalysisConfig(
            options.prefix,
            "\\|".join(options.filter),
            options.reps,
            options.retries,
            options.dry_run,
            options.mhp_bench,
            options.shp_bench,
            options.weak_scaling,
            options.different_devices,
            options.device_memory,
        )
    )
    for t in options.target:
        for s in options.vec_size:
            for n in options.ranks:
                click.echo(f"::group::dr-bench run ranks: {n} target: {t}")
                r.run_one_analysis(
                    runner.AnalysisCase(choice_to_target(t), s, n, options.ppn)
                )
                click.echo("::endgoup::")


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
@option_reps
@option_retries
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
@click.option(
    "--weak-scaling",
    is_flag=True,
    default=False,
    help="Scales the vector size by the number of ranks",
)
@option_different_devices
def run(
    prefix,
    target,
    vec_size,
    ranks,
    rank_range,
    reps,
    retries,
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

    options.device_memory = device_memory
    options.prefix = prefix
    options.target = target
    options.vec_size = vec_size
    options.ranks = ranks
    if rank_range:
        options.ranks = list(range(1, rank_range + 1))
    options.reps = reps
    options.retries = retries
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
@click.option(
    "--vec-size",
    type=int,
    default=2000000000,
    help="Size of a vector",
)
@option_reps
@option_retries
@click.option(
    "--min-gpus",
    type=int,
    default=1,
    help="Beginning of range of GPUs",
)
@click.option(
    "--gpus",
    type=int,
    default=0,
    help="End of range of GPUs",
)
@option_ppn
@option_different_devices
@click.option(
    "--mhp-only",
    is_flag=True,
    default=False,
    help="Do not run shp or reference",
)
def suite(
    prefix,
    mhp_bench,
    shp_bench,
    dry_run,
    clean,
    vec_size,
    reps,
    retries,
    min_gpus,
    gpus,
    ppn,
    different_devices,
    mhp_only,
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
        options.ppn = ppn
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

    # Run subsequence of 1, 2, 4, 8, 12, ...
    def run_rank_sparse(
        base,
        min_ranks,
        max_ranks,
        filters,
        targets,
        weak_scaling_filter=[],
        device_memory=False,
    ):
        run_rank_list(
            base,
            list(
                filter(
                    lambda r: r >= min_ranks and r <= max_ranks,
                    [1, 2, 4, 8, 12, 24, 48, 96],
                )
            ),
            filters,
            targets,
            weak_scaling_filter=weak_scaling_filter,
            device_memory=device_memory,
        )

    def run_mhp(base):
        #
        # GPU devices
        #
        if gpus > 0:
            # benchmarks
            run_rank_sparse(
                base,
                min_gpus,
                gpus,
                xhp_filter + mhp_filter,
                ["mhp_sycl_gpu"],
                weak_scaling_filter,
            )
            run_rank_sparse(
                base,
                min_gpus,
                gpus,
                device_memory_filter,
                ["mhp_sycl_gpu"],
                device_memory=True,
            )

    def run_shp(base):
        #
        # GPU devices
        #
        if gpus > 0:
            # benchmarks
            run_rank_sparse(
                base,
                min_gpus,
                gpus,
                xhp_filter + shp_filter,
                ["shp_sycl_gpu"],
                weak_scaling_filter,
                device_memory=True,
            )

            run_rank_sparse(
                base,
                min_gpus,
                min(gpus, 4),
                shp_no_more_than_4_filter,
                ["shp_sycl_gpu"],
                weak_scaling_filter,
                device_memory=True,
            )

    def run_reference(base):
        #
        # GPU devices
        #
        if gpus > 0:
            # reference
            run_rank_sparse(base, 1, 1, mhp_reference_filter, ["mhp_sycl_gpu"])
            run_rank_sparse(
                base,
                1,
                1,
                sycl_reference_filter + shp_reference_filter,
                ["shp_sycl_gpu"],
            )

    # benchmark filters
    xhp_filter = [
        "BlackScholes_DR",
        "DotProduct_DR",
        # does not work
        # "Exclusive_Scan_DR"
        "Inclusive_Scan_DR",
        "Reduce_DR",
        "Stream_Triad",
    ]
    weak_scaling_filter = [
        "Inclusive_Scan_DR",
        "Reduce_DR",
        ".*Sort_DR",
        "WaveEquation_DR",
        "Gemm_Reference",
    ]
    mhp_filter = ["Stencil2D_DR", "WaveEquation_DR"]
    device_memory_filter = ["FFT3D_DR", "WaveEquation_DR"]
    shp_filter = [".*Sort_DR", "Gemm_DR"]
    # FFT3D_DR fails with PI_OUT_OF_RESOURCES when GPUS>4,
    # fails always on 2024.1 and sometimes on 2023.2
    shp_no_more_than_4_filter = ["FFT3D_DR"]
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

    # Base options
    base = SuiteConfig()
    base.prefix = prefix
    base.mhp_bench = mhp_bench
    base.shp_bench = shp_bench
    base.dry_run = dry_run
    base.vec_size = [vec_size]
    base.reps = reps
    base.retries = retries
    base.different_devices = different_devices

    logging.info(f"different_devices:{different_devices}")

    if clean and not dry_run:
        do_clean(prefix)

    run_mhp(base)
    if not mhp_only:
        run_shp(base)
        run_reference(base)


if __name__ == "__main__":
    assert False  # not to be used this way, but by dr-bench executable
