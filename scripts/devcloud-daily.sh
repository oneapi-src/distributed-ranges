#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
source /opt/intel/oneapi/setvars.sh
set -e
hostname
export CXX=icpx 
cmake -DENABLE_SYCL:BOOL=ON -DENABLE_SYCL_MPI:BOOL=ON -B build -S .
make -C build -j

# python -m pip install --upgrade pip
cd build
# pip install -r requirements.txt
python3 ../benchmarks/runner/run_benchmarks.py