#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# system bashrc does this
# source /opt/intel/oneapi/setvars.sh

set -e
hostname

# devcloud requires --launcher=fork for mpi
cmake -B build -DENABLE_SYCL=on -DENABLE_MPIFORK=on

# default sycl device will be GPU, if available. Only use 1 GPU:
# workaround for devcloud multi-card not working
ONEAPI_DEVICE_SELECTOR=level_zero:0 make -j -C build all test
# another run, forcing CPU
ONEAPI_DEVICE_SELECTOR=opencl:cpu make -j -C build all test
