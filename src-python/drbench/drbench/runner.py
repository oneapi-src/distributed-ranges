# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import resource
import subprocess
import sys
import uuid
from collections import namedtuple

from drbench.common import Device, Model, Runtime

AnalysisCase = namedtuple("AnalysisCase", "target size ranks ppn")
AnalysisConfig = namedtuple(
    "AnalysisConfig",
    " ".join(
        [
            "prefix",
            "benchmark_filter",
            "reps",
            "retries",
            "dry_run",
            "mhp_bench",
            "shp_bench",
            "weak_scaling",
            "different_devices",
            "device_memory",
        ]
    ),
)


class Runner:
    def __init__(self, analysis_config: AnalysisConfig):
        self.analysis_config = analysis_config

    def __execute(self, command: str):
        runs_count = 0
        while True:
            usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            runs_count += 1
            try:
                logging.info(
                    f"execute {runs_count}/{self.analysis_config.retries}\n"
                    f"  {command}"
                )
                if not self.analysis_config.dry_run:
                    subprocess.run(
                        command, shell=True, check=True, timeout=300
                    )
                usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
                logging.info(
                    f" result: command execution time:"
                    f" user:{usage_end.ru_utime - usage_start.ru_utime:.2f}, "
                    f" system:{usage_end.ru_stime - usage_start.ru_stime:.2f}"
                )
                break
            except subprocess.TimeoutExpired:
                logging.warning("command timed out")
            except subprocess.CalledProcessError as exc:
                logging.warning(f"command failed with code:{exc.returncode}")

            if runs_count > self.analysis_config.retries:
                logging.error("failed too mamy times, no more retries")
                sys.exit(1)

    def __run_mhp_analysis(self, params, ranks, ppn, target):
        if target.runtime == Runtime.SYCL:
            params.append("--sycl")
            if target.device == Device.CPU:
                env = "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
            else:
                env = (
                    "ONEAPI_DEVICE_SELECTOR="
                    "'level_zero:gpu;ext_oneapi_cuda:gpu'"
                    # GPU aware MPI
                    " I_MPI_OFFLOAD=1"
                    # tile i assigned to rank i
                    " I_MPI_OFFLOAD_CELL_LIST=0-11"
                    # do not use the SLURM/PBS resource manager to launch jobs
                    " I_MPI_HYDRA_BOOTSTRAP=ssh"
                )
        else:
            env = (
                "I_MPI_PIN_DOMAIN=core "
                "I_MPI_PIN_ORDER=compact "
                "I_MPI_PIN_CELL=unit"
            )

        self.__execute(
            # mpiexec bypasses SLURM/PBS configuration
            f"{env} {os.environ['I_MPI_ROOT']}/bin/mpiexec.hydra -n {ranks} "
            f"-ppn {ppn} {self.analysis_config.mhp_bench} {' '.join(params)}"
        )

    def __run_shp_analysis(self, params, ranks, target):
        if target.device == Device.CPU:
            env = "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
        else:
            env = (
                "ONEAPI_DEVICE_SELECTOR="
                "'level_zero:gpu;ext_oneapi_cuda:gpu'"
            )
        env += " KMP_AFFINITY=compact"
        params.append(f"--num-devices {ranks}")
        self.__execute(
            f'{env} {self.analysis_config.shp_bench} {" ".join(params)}'
        )

    def run_one_analysis(self, analysis_case: AnalysisCase):
        params = []
        target = analysis_case.target
        params.append(f"--vector-size {str(analysis_case.size)}")
        params.append(f"--reps {str(self.analysis_config.reps)}")
        params.append("--benchmark_out_format=json")
        params.append(f"--context device:{target.device.name}")
        params.append(f"--context model:{target.model.name}")
        params.append(f"--context runtime:{target.runtime.name}")
        params.append(f"--context target:{target}")
        params.append("--v=3")  # verbosity

        prefix = self.analysis_config.prefix
        params.append(f"--benchmark_out={prefix}-{uuid.uuid4().hex}.json")

        if self.analysis_config.benchmark_filter:
            params.append(
                f"--benchmark_filter={self.analysis_config.benchmark_filter}"
            )

        if self.analysis_config.weak_scaling:
            params.append("--weak-scaling")

        if self.analysis_config.device_memory:
            params.append("--device-memory")

        if self.analysis_config.different_devices:
            params.append("--different-devices")

        if analysis_case.target.model == Model.SHP:
            self.__run_shp_analysis(
                params, analysis_case.ranks, analysis_case.target
            )
        else:
            self.__run_mhp_analysis(
                params,
                analysis_case.ranks,
                analysis_case.ppn,
                analysis_case.target,
            )
