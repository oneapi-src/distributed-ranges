#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -e
hostname
source /opt/intel/oneapi/setvars.sh
# devcloud requires --launcher=fork for mpi
cmake -B build -DENABLE_SYCL=on -DENABLE_MPIFORK=on

BUILD_TARGETS=("$@")
cmake --build build -j "${BUILD_TARGETS[@]/#/-t }"
