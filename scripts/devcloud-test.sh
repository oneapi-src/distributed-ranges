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
unset ONEAPI_DEVICE_SELECTOR

echo "::group::Generate"
time cmake -B build -DENABLE_SYCL=on
echo "::endgroup::"

echo "::group::Build"
time make -C build all -j
echo "::endgroup::"

echo "::group::SHP GPU Test"
# Use 1 device because p2p does not work
ONEAPI_DEVICE_SELECTOR=level_zero:0 time ctest --test-dir build -L SHP
echo "::endgroup::"

# disabled: very slow or fails when cryptominer is on devcloud
#echo "::group::SHP CPU Test"
#ONEAPI_DEVICE_SELECTOR=opencl:cpu time ctest --test-dir build -L SHP
#echo "::endgroup::"

echo "::group::MP GPU Test"
ONEAPI_DEVICE_SELECTOR=level_zero:* time ctest --test-dir build -L MP
echo "::endgroup::"

echo "::group::MP CPU Test"
ONEAPI_DEVICE_SELECTOR=opencl:cpu time ctest --test-dir build -L MP
echo "::endgroup::"
