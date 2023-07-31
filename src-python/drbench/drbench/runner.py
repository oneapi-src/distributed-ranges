# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import subprocess
from collections import namedtuple

import click
from drbench import common
from drbench.common import Device, Model, Runtime

AnalysisCase = namedtuple("AnalysisCase", "target size nprocs")
AnalysisConfig = namedtuple(
    "AnalysisConfig",
    "common_config benchmark_filter fork reps dry_run mhp_bench shp_bench",
)


class Runner:
    def __init__(self, analysis_config: AnalysisConfig):
        self.analysis_config = analysis_config

    def __execute(self, command: str):
        click.echo(command)
        if not self.analysis_config.dry_run:
            subprocess.run(command, shell=True, check=True)

    def __out_filename(self, case: AnalysisCase, add_nnn: bool):
        prefix = common.analysis_file_prefix(
            self.analysis_config.common_config.analysis_id
        )

        i = 0
        while True:
            p = f"{prefix}.{case.target}.n{case.nprocs}.s{case.size}.i{i}i"
            if not glob.glob(f"{p}*"):
                rank = ".rankNNN" if add_nnn else ""
                return f"{p}{rank}.json"
            i = i + 1

    def __run_mhp_analysis(self, params, nprocs, target):
        if target.runtime == Runtime.SYCL:
            params.append("--sycl")
            if target.device == Device.CPU:
                env = "ONEAPI_DEVICE_SELECTOR=opencl:cpu"
            else:
                env = (
                    "ONEAPI_DEVICE_SELECTOR="
                    "'level_zero:gpu;ext_oneapi_cuda:gpu'"
                )
        else:
            env = (
                "I_MPI_PIN_DOMAIN=core "
                "I_MPI_PIN_ORDER=compact "
                "I_MPI_PIN_CELL=unit"
            )

        mpirun_params = []
        mpirun_params.append(f"-n {str(nprocs)}")
        if self.analysis_config.fork:
            mpirun_params.append("-launcher=fork")

        self.__execute(
            env
            + " mpirun "
            + " ".join(mpirun_params)
            + " "
            + self.analysis_config.mhp_bench
            + " "
            + " ".join(params)
        )

    def __run_shp_analysis(self, params, nprocs):
        env = "KMP_AFFINITY=compact"
        params.append(f"--num-devices {nprocs}")
        self.__execute(
            f'{env} {self.analysis_config.shp_bench} {" ".join(params)}'
        )

    def run_one_analysis(self, analysis_case: AnalysisCase):
        params = []
        params.append(f"--vector-size {str(analysis_case.size)}")
        params.append(f"--reps {str(self.analysis_config.reps)}")
        params.append("--benchmark_out_format=json")
        outfname = self.__out_filename(
            analysis_case, analysis_case.target.model != Model.SHP
        )
        params.append(f"--benchmark_out={outfname}")

        if self.analysis_config.benchmark_filter:
            params.append(
                f"--benchmark_filter={self.analysis_config.benchmark_filter}"
            )

        if analysis_case.target.model == Model.SHP:
            self.__run_shp_analysis(params, analysis_case.nprocs)
        else:
            self.__run_mhp_analysis(
                params, analysis_case.nprocs, analysis_case.target
            )
