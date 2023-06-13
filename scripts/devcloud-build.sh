#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -xe
source /opt/intel/oneapi/setvars.sh
cmake -B build -DENABLE_SYCL=on
make -C build/test/gtest/shp -j
make -C build/test/gtest/mhp -j
