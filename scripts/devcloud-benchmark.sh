#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


# system bashrc does this
# source /opt/intel/oneapi/setvars.sh

set -e
hostname
cmake -B build -DENABLE_SYCL=on
cmake --build build -j --target devcloud-bench
