#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
set -e
hostname
source /opt/intel/oneapi/setvars.sh
cmake -DENABLE_SYCL:BOOL=ON -DENABLE_SYCL_MPI:BOOL=ON -B build -S .
make -C build -j

# python -m pip install --upgrade pip
cd benchmarks/runner
# pip install -r requirements.txt
python3 run_benchmarks.py