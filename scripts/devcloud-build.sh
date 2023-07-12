#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -e
hostname
source /opt/intel/oneapi/setvars.sh
cmake -B build -DENABLE_SYCL=on
make -C build -j
