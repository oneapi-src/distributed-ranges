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

cmake -B build -DENABLE_SYCL=on

# default sycl device will be GPU, if available. Only use 1 GPU:
# workaround for devcloud multi-card not working
export ONEAPI_DEVICE_SELECTOR=level_zero:0
cd build/test/gtest/mhp
date
make -j mhp-tests
date
ctest -VV -R '.*device.*'
