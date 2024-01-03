#!/bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
set -e

# needed for tasks running under PBS
cd ${PBS_O_WORKDIR:-.}

# needed for tasks running under SLURM, as SLURM/MPI integration is broken
unset SLURM_TASKS_PER_NODE
unset SLURM_JOBID

echo "Host: " $(hostname)
echo "CWD: " $(pwd)
module list

if [ "$#" -le 1 ]; then
	echo "provide as first argument a log file"
	exit 1
fi

LOGFILE=$1
mkdir -p $(dirname $LOGFILE)

shift 1

"$@" 2>&1 | tee $LOGFILE
