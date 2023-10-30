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
unset ONEAPI_DEVICE_SELECTOR

echo ***Generate***
time cmake -B build -DENABLE_SYCL=on

echo ***Build***
time make -C build all

echo ***SHP GPU Test***
# Use 1 device because p2p does not work
ONEAPI_DEVICE_SELECTOR=level_zero:0 time ctest -B build -L SHP

echo ***SHP CPU Test***
# Use 1 device because p2p does not work
ONEAPI_DEVICE_SELECTOR=opencl:cpu time ctest -B build -L SHP

echo ***MHP GPU Test***
# Use 1 device because p2p does not work
ONEAPI_DEVICE_SELECTOR=level_zero:* time ctest -B build -L MHP

echo ***MHP CPU Test***
# Use 1 device because p2p does not work
ONEAPI_DEVICE_SELECTOR=opencl:cpu time ctest -B build -L MHP
