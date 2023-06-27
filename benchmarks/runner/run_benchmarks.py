#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import subprocess
import json
import os
import datetime


def run_mhp(
    vec_size: int,
    reps: int,
    filter: str,
    sycl_used: bool,
    only_fsycl: bool,
    mpirun: bool = True,
    n: int = None,
    pin_domain: str = None,
    pin_order: str = None,
    pin_cell: str = None,
):
    os.environ["BENCHMARK_FILTER"] = filter
    if pin_domain is not None:
        os.environ["I_MPI_PIN_DOMAIN"] = pin_domain
    if pin_order is not None:
        os.environ["I_MPI_PIN_ORDER="] = pin_order
    if pin_cell is not None:
        os.environ["I_MPI_PIN_CELL="] = pin_cell
    if only_fsycl:
        bench_path = "./build/benchmarks/gbench/mhp/mhp-bench-only-fsycl"
    else:
        bench_path = "./build/benchmarks/gbench/mhp/mhp-bench"

    time_now = datetime.datetime.now()
    time_now_string = datetime.datetime(
        time_now.year,
        time_now.month,
        time_now.day,
        time_now.hour,
        time_now.minute,
    ).isoformat()
    bench_out = f"--benchmark_out=benchmark_{time_now_string}.json --benchmark_out_format=json"
    if n == None:
        n = ""
    else:
        n = f"-n {str(n)}"
    if mpirun:
        mpirun = "mpirun"
    else:
        mpirun = ""
        n = ""
    if sycl_used:
        command = [
            mpirun,
            n,
            bench_path,
            "--sycl",
            "--vector-size",
            str(vec_size),
            "--reps",
            str(reps),
            bench_out,
        ]
    else:
        command = [
            mpirun,
            n,
            bench_path,
            "--vector-size",
            str(vec_size),
            "--reps",
            str(reps),
            bench_out,
        ]
    while "" in command:
        command.remove('')
    command = " ".join(command)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = process.communicate()
    print(output)
    print(err)


def run_shp(
    vec_size: int,
    reps: int,
    d: int,
    only_fsycl: bool,
    filter: str = None,
    kmp_aff: str = None,
):
    if filter is not None:
        os.environ["BENCHMARK_FILTER"] = filter
    else:
        try:
            os.unsetenv("BENCHMARK_FILTER")
        except:
            pass
    if kmp_aff is not None:
        os.environ["KMP_AFFINITY"] = kmp_aff
    if only_fsycl:
        bench_path = "./build/benchmarks/gbench/shp/shp-bench-only-fsycl"
    else:
        bench_path = "./build/benchmarks/gbench/shp/shp-bench"
    bench_out = "--benchmark_out=test2.json --benchmark_out_format=json"
    command = [
        bench_path,
        "-d",
        str(d),
        "--vector-size",
        str(vec_size),
        "--reps",
        str(reps),
        bench_out,
    ]
    while "" in command:
        command.remove('')
    command = " ".join(command)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = process.communicate()


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
