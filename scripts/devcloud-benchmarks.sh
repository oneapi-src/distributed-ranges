#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
source /opt/intel/oneapi/setvars.sh
set -e
hostname
cmake --build build --target bench
