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

@click.command()
@click.option('--dry_run', default=False, help='Emits commands but does not execute')
def print_run_command(command:str, dry_run:bool):
    print(f"Current benchmark command used: \n{command}")
    if not dry_run:
        output = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

def run_mhp(
    vec_size: int,
    reps: int,
    filter: str,
    sycl_used: bool,
    only_fsycl: bool,
    mpirun: bool = True,
    n: int = None,
    pin_domain: str = '',
    pin_order: str = '',
    pin_cell: str = '',
):
    if only_fsycl:
        bench_path = "./build/benchmarks/gbench/mhp/mhp-bench-only-fsycl"
    else:
        bench_path = "./build/benchmarks/gbench/mhp/mhp-bench"

    time_now_string = datetime.datetime.now().isoformat(timespec='minutes')
    directory_path = f"build/benchmarks/json_output/mhp"
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
        command = f"BENCHMARK_FILTER={filter} I_MPI_PIN_DOMAIN={pin_domain} I_MPI_PIN_ORDER={pin_order} I_MPI_PIN_CELL={pin_cell} {mpirun} {n} {bench_path} --sycl --vector-size {str(vec_size)} --reps {str(reps)} {bench_out}"
    else:
        command = f"BENCHMARK_FILTER={filter} I_MPI_PIN_DOMAIN={pin_domain} I_MPI_PIN_ORDER={pin_order} I_MPI_PIN_CELL={pin_cell} {mpirun} {n} {bench_path} --vector-size {str(vec_size)} --reps {str(reps)} {bench_out}"

    print_run_command(command)


def run_shp(
    vec_size: int,
    reps: int,
    d: int,
    only_fsycl: bool,
    filter: str = '',
    kmp_aff: str = '',
):
    if only_fsycl:
        bench_path = "./build/benchmarks/gbench/shp/shp-bench-only-fsycl"
    else:
        bench_path = "./build/benchmarks/gbench/shp/shp-bench"
    time_now_string = datetime.datetime.now().isoformat(timespec='minutes')
    directory_path = f"build/benchmarks/json_output/shp"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    bench_out = f"--benchmark_out={directory_path}/shp_benchmark_{time_now_string}.json --benchmark_out_format=json"
    command = f"BENCHMARK_FILTER={filter} KMP_AFFINITY={kmp_aff} {bench_path} -d {str(d)} --vector-size {str(vec_size)} --reps {str(reps)} {bench_out}"
    print_run_command(command)


def run_all_benchmarks_fsycl_O3():
    vec_size = 1000000000
    reps = 100
    bench_filter = "Stream_"
    kmp_affinity = "compact"
    only_fsycl = True

    # shp
    d = 1
    run_shp(vec_size, reps, d, only_fsycl)
    run_shp(vec_size, reps, d, only_fsycl, bench_filter, kmp_affinity)

    d = 2
    run_shp(vec_size, reps, d, only_fsycl)
    run_shp(vec_size, reps, d, only_fsycl, bench_filter, kmp_affinity)

    # mhp/cpu
    sycl_used = False
    mpirun = False
    run_mhp(vec_size, reps, bench_filter, sycl_used, mpirun=False)
    n = 24
    mpi_pin_domain = "core"
    mpi_pin_order = "compact"
    mpi_pin_cell = "unit"

    mpirun = True
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        mpirun,
        n,
        mpi_pin_domain,
        mpi_pin_order,
        mpi_pin_cell,
    )
    n = 48
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        mpirun,
        n,
        mpi_pin_domain,
        mpi_pin_order,
        mpi_pin_cell,
    )

    # mhp/sycl
    mpi_pin_domain = "socket"
    sycl_used = True
    n = 1
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        n,
        mpi_pin_domain,
        mpirun=mpirun,
    )
    run_mhp(vec_size, reps, bench_filter, sycl_used, n, mpirun=mpirun)
    n = 2
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        n,
        mpi_pin_domain,
        mpi_pin_order,
        mpi_pin_cell,
        mpirun=mpirun,
    )


def run_all_benchmarks_other_options():
    vec_size = 1000000000
    reps = 100
    bench_filter = "Stream_"
    kmp_affinity = "compact"
    only_fsycl = False
    mpirun = True
    # shp
    d = 1
    run_shp(vec_size, reps, d, only_fsycl, bench_filter, kmp_affinity)
    d = 2
    run_shp(vec_size, reps, d, only_fsycl, bench_filter, kmp_affinity)
    # mhp/cpu
    sycl_used = False
    mpi_pin_domain = "core"
    mpi_pin_order = "compact"
    mpi_pin_cell = "unit"
    n = 24
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        only_fsycl,
        mpirun,
        n,
        mpi_pin_domain,
        mpi_pin_order,
        mpi_pin_cell,
    )
    n = 48
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        only_fsycl,
        mpirun,
        n,
        mpi_pin_domain,
        mpi_pin_order,
        mpi_pin_cell,
    )


# def save_to_csv(self, output):
#     rows = output.splitlines()
#     rows = [row.split(',') for row in rows]

#     with open(self.csv_path, 'a', newline='') as f:
#         writer = csv.writer(f)
#         for row in rows:
#             writer.writerow(row)

# def extract_json_save_csv(filepath:str):
#     file = json.load(filepath)


def run_mhp_test():
    vec_size = 1000000000
    reps = 100
    bench_filter = "Stream_"
    kmp_affinity = "compact"
    only_fsycl = True
    sycl_used = False
    mpirun = False
    n = 1
    run_mhp(
        vec_size,
        reps,
        bench_filter,
        sycl_used,
        only_fsycl,
        mpirun,
        n,
    )


def run_shp_test():
    vec_size = 1000000000
    reps = 100
    d = 1
    only_fsycl = True

    run_shp(vec_size, reps, d, only_fsycl)


run_mhp_test()
run_shp_test()
