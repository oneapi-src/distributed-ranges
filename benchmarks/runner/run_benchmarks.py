#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import subprocess
import json
import os
import datetime
import click

def bench_common_params(func):
    func = click.option('--vec_size', default=1000000, type=int, help='Size of a vector')(func)
    func = click.option('--reps', default=100, type=int, help='Number of reps')(func)
    func = click.option('--bench_filter', default='Stream_', type=str, help='A filter used for a benchmark')(func)
    return func

@click.command()
@click.option('--dry_run', default=False, help='Emits commands but does not execute')
def print_run_command(command:str, dry_run:bool):
    click.echo(f"Current benchmark command used: \n{command}")
    if not dry_run:
        output = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

@click.command()
@bench_common_params
def run_mhp(
    vec_size: int,
    reps: int,
    bench_filter: str,
    sycl_used: bool,
    mpirun: bool = True,
    n: int = None,
    pin_domain: str = '',
    pin_order: str = '',
    pin_cell: str = '',
):
    bench_path = "./benchmarks/gbench/mhp/mhp-bench"
    time_now_string = datetime.datetime.now().isoformat(timespec='minutes')
    directory_path = f"benchmarks/json_output/mhp"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    bench_out = f"--benchmark_out={directory_path}/mhp_benchmark_{time_now_string}.json --benchmark_out_format=json"
    if n is None:
        n = ""
    else:
        n = f"-n {str(n)}"
    if mpirun:
        mpirun = "mpirun"
    else:
        mpirun = ""
        n = ""
    if sycl_used:
        command = f"BENCHMARK_FILTER={bench_filter} I_MPI_PIN_DOMAIN={pin_domain} I_MPI_PIN_ORDER={pin_order} I_MPI_PIN_CELL={pin_cell} {mpirun} {n} {bench_path} --sycl --vector-size {str(vec_size)} --reps {str(reps)} {bench_out}"
    else:
        command = f"BENCHMARK_FILTER={bench_filter} I_MPI_PIN_DOMAIN={pin_domain} I_MPI_PIN_ORDER={pin_order} I_MPI_PIN_CELL={pin_cell} {mpirun} {n} {bench_path} --vector-size {str(vec_size)} --reps {str(reps)} {bench_out}"

    print_run_command(command)

@click.command()
@bench_common_params
def run_shp(
    vec_size: int,
    reps: int,
    d: int,
    bench_filter: str = '',
    kmp_aff: str = '',
):
    bench_path = "./benchmarks/gbench/shp/shp-bench"
    time_now_string = datetime.datetime.now().isoformat(timespec='minutes')
    directory_path = f"benchmarks/json_output/shp"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    bench_out = f"--benchmark_out={directory_path}/shp_benchmark_{time_now_string}.json --benchmark_out_format=json"
    command = f"BENCHMARK_FILTER={bench_filter} KMP_AFFINITY={kmp_aff} {bench_path} -d {str(d)} --vector-size {str(vec_size)} --reps {str(reps)} {bench_out}"
    print_run_command(command)

def run_all_benchmarks_fsycl_O3():
    kmp_affinity = "compact"

    # shp
    d = 1
    run_shp(d=d)
    run_shp(d=d, kmp_aff=kmp_affinity)

    d = 2
    run_shp(d=d)
    run_shp(d=d, kmp_aff=kmp_affinity)

    # mhp/cpu
    sycl_used = False
    mpirun = False
    run_mhp(sycl_used=sycl_used, mpirun=False)
    n = 24
    mpi_pin_domain = "core"
    mpi_pin_order = "compact"
    mpi_pin_cell = "unit"

    mpirun = True
    run_mhp(
        sycl_used=sycl_used,
        mpirun=mpirun,
        n=n,
        pin_domain=mpi_pin_domain,
        pin_order=mpi_pin_order,
        pin_cell=mpi_pin_cell,
    )
    n = 48
    run_mhp(
        sycl_used=sycl_used,
        mpirun=mpirun,
        n=n,
        pin_domain=mpi_pin_domain,
        pin_order=mpi_pin_order,
        pin_cell=mpi_pin_cell,
    )

    # mhp/sycl
    mpi_pin_domain = "socket"
    sycl_used = True
    n = 1
    run_mhp(
        sycl_used=sycl_used,
        n=n,
        pin_domain=mpi_pin_domain,
        mpirun=mpirun,
    )
    run_mhp(sycl_used=sycl_used, n=n, mpirun=mpirun)
    n = 2
    run_mhp(
        sycl_used=sycl_used,
        n=n,
        pin_domain=mpi_pin_domain,
        pin_order=mpi_pin_order,
        pin_cell=mpi_pin_cell,
        mpirun=mpirun,
    )

def run_all_benchmarks_other_options():
    kmp_affinity = "compact"
    mpirun = True
    # shp
    d = 1
    run_shp(d=d, kmp_aff=kmp_affinity)
    d = 2
    run_shp(d=d, kmp_aff=kmp_affinity)
    # mhp/cpu
    sycl_used = False
    mpi_pin_domain = "core"
    mpi_pin_order = "compact"
    mpi_pin_cell = "unit"
    n = 24
    run_mhp(
        sycl_used=sycl_used,
        mpirun=mpirun,
        n=n,
        pin_domain=mpi_pin_domain,
        pin_order=mpi_pin_order,
        pin_cell=mpi_pin_cell,
    )
    n = 48
    run_mhp(
        sycl_used=sycl_used,
        mpirun=mpirun,
        n=n,
        pin_domain=mpi_pin_domain,
        pin_order=mpi_pin_order,
        pin_cell=mpi_pin_cell,
    )

def run_mhp_test():
    sycl_used = False
    mpirun = False
    n = 1
    run_mhp(
        sycl_used=sycl_used,
        mpirun=mpirun,
        n=n,
    )


def run_shp_test():
    d = 1
    run_shp(d=d)


run_mhp_test()
run_shp_test()
