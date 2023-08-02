#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
source /opt/intel/oneapi/setvars.sh
set -e
hostname
# devcloud requires --launcher=fork for mpi
cmake -B build -DENABLE_SYCL=on -DENABLE_MPIFORK=on
cmake --build build --target devcloud-bench
