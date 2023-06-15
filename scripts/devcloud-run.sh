#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set -xe
source /opt/intel/oneapi/setvars.sh
make -C build test
