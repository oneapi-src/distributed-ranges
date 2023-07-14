#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -e
hostname
source /opt/intel/oneapi/setvars.sh

# we clear the environment to make mpi work, add back error output
export CTEST_OUTPUT_ON_FAILURE=1

# default sycl device will be GPU, if available. Only use 1 GPU:
# workaround for devcloud multi-card not working
ONEAPI_DEVICE_SELECTOR=level_zero:0 make -C build test
# another run, forcing CPU
ONEAPI_DEVICE_SELECTOR=opencl:cpu make -C build test
