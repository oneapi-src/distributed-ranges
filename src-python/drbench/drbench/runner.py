# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO: more stuff is going to be moved here from drbench.py

import subprocess
from collections import namedtuple
from enum import Enum

import click
from drbench import common

AnalysisMode = Enum(
    'AnalysisMode', ['MHP_CPU', 'MHP_GPU', 'MHP_NOSYCL', 'SHP']
)
AnalysisCase = namedtuple('AnalysisCase', 'mode size nprocs')
AnalysisConfig = namedtuple(
    'AnalysisConfig',
    'common_config benchmark_filter fork reps dry_run mhp_bench shp_bench',
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
        rank = ".rankNNN" if add_nnn else ""
        return f'{prefix}{rank}.{case.mode}.n{case.nprocs}.s{case.size}.json'

    def __run_mhp_analysis(self, params, nprocs, mode):
        if mode == AnalysisMode.MHP_CPU:
            env = 'ONEAPI_DEVICE_SELECTOR=opencl:cpu'
            params.append('--sycl')
        elif mode == AnalysisMode.MHP_GPU:
            env = (
                'ONEAPI_DEVICE_SELECTOR=\'level_zero:gpu;ext_oneapi_cuda:gpu\''
            )
            params.append('--sycl')
        elif mode == AnalysisMode.MHP_NOSYCL:
            env = (
                'I_MPI_PIN_DOMAIN=core '
                'I_MPI_PIN_ORDER=compact '
                'I_MPI_PIN_CELL=unit'
            )
        else:
            assert False

        mpirun_params = []
        mpirun_params.append(f'-n {str(nprocs)}')
        if self.analysis_config.fork:
            mpirun_params.append('-launcher=fork')

        self.__execute(
            env
            + ' mpirun '
            + " ".join(mpirun_params)
            + ' '
            + self.analysis_config.mhp_bench
            + ' '
            + " ".join(params)
        )

    def __run_shp_analysis(self, params, nprocs):
        env = 'KMP_AFFINITY=compact'
        params.append(f'--num-devices {nprocs}')
        self.__execute(
            f'{env} {self.analysis_config.shp_bench} {" ".join(params)}'
        )

    def run_one_analysis(self, analysis_case: AnalysisCase):
        params = []
        params.append(f'--vector-size {str(analysis_case.size)}')
        params.append(f'--reps {str(self.analysis_config.reps)}')
        params.append('--benchmark_out_format=json')
        outfname = self.__out_filename(
            analysis_case, analysis_case.mode != AnalysisMode.SHP
        )
        params.append(f'--benchmark_out={outfname}')

        if self.analysis_config.benchmark_filter:
            params.append(
                f'--benchmark_filter={self.analysis_config.benchmark_filter}'
            )

        if analysis_case.mode == AnalysisMode.SHP:
            self.__run_shp_analysis(params, analysis_case.nprocs)
        else:
            self.__run_mhp_analysis(
                params, analysis_case.nprocs, analysis_case.mode
            )
