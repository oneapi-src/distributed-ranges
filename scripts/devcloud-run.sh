#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -e
hostname
source /opt/intel/oneapi/setvars.sh
# default sycl device will be GPU, if available
export CTEST_OUTPUT_ON_FAILURE=1
make -C build test
# another run, forcing CPU
ONEAPI_DEVICE_SELECTOR=opencl:cpu make -C build test
