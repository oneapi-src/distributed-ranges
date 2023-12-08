#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
set -e

cd $PBS_O_WORKDIR

echo "Host: " $(hostname)
echo "CWD: " $(pwd)
module list

source venv/bin/activate

printenv > build/envdump.txt

# Builds dependencies and runs benchmarks
cmake --build build -j --target aurora-bench > build/cmake-output-1.txt 2>&1
