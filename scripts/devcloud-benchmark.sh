#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

source scripts/setvars-2023.2.1.sh
set -e
hostname

# SLURM/MPI integration is broken
unset SLURM_TASKS_PER_NODE
unset SLURM_JOBID

cmake -B build -DENABLE_SYCL=on
cmake --build build -j --target devcloud-bench
