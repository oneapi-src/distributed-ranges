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
    help="Number of processes",
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
@click.option(
    "--mhp-bench",
    default="mhp/mhp-bench",
    type=str,
    help="MHP benchmark program",
)
@click.option(
    "--shp-bench",
    default="shp/shp-bench",
    type=str,
    help="SHP benchmark program",
)
@click.option(
    "-d", "--dry-run", is_flag=True, help="Emits commands but does not execute"
)
@click.option(
    "-c", "--clean", is_flag=True, help="Delete all json files with the prefix"
)
def run(
    prefix,
    target,
    vec_size,
    ranks,
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

    if clean:
        do_clean(prefix)

    r = runner.Runner(
        runner.AnalysisConfig(
            prefix,
            "\\|".join(filter),
            reps,
            dry_run,
            mhp_bench,
            shp_bench,
        )
    )
    click.echo(f"Targets: {target}")
    click.echo(f"Ranks: {ranks}")
    for t in target:
        for s in vec_size:
            for n in ranks:
                r.run_one_analysis(
                    runner.AnalysisCase(choice_to_target(t), s, n)
                )


if __name__ == "__main__":
    assert False  # not to be used this way, but by dr-bench executable
