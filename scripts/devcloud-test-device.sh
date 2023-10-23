#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

source /opt/intel/oneapi/setvars.sh
set -e
hostname

# SLURM/MPI integration is broken
unset SLURM_TASKS_PER_NODE
unset SLURM_JOBID

cmake -B build -DENABLE_SYCL=on -DENABLE_DEVICE_MEMORY=on
make -j -C build/test/gtest/mhp all test
