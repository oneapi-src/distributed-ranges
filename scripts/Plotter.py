#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import re
from dataclasses import dataclass

import click
import matplotlib.pyplot as plt


@dataclass
class BenchResults:
    copy: float
    scale: float
    add: float
    triad: float


class Plotter:
    def __init__(self) -> None:
        self.plots_path_shp = "scripts"
        self.plots_path_mhp = "scripts"
        self.bench_path_shp = "scripts"
        self.bench_path_mhp = "scripts"
        self.create_directories()

    def create_directories(self):
        if not os.path.exists(self.plots_path_shp):
            os.makedirs(self.plots_path_shp)
        if not os.path.exists(self.plots_path_mhp):
            os.makedirs(self.plots_path_mhp)
        if not os.path.exists(self.bench_path_shp):
            os.makedirs(self.bench_path_shp)
        if not os.path.exists(self.bench_path_mhp):
            os.makedirs(self.bench_path_mhp)

    def clean_directories(self):
        if not os.path.exists(self.plots_path_shp):
            for file in os.listdir(self.plots_path_shp):
                os.remove(os.path.join(self.plots_path_shp), file)
        if not os.path.exists(self.plots_path_mhp):
            for file in os.listdir(self.plots_path_mhp):
                os.remove(os.path.join(self.plots_path_mhp), file)
        if not os.path.exists(self.bench_path_shp):
            for file in os.listdir(self.bench_path_shp):
                os.remove(os.path.join(self.bench_path_shp), file)
        if not os.path.exists(self.bench_path_mhp):
            for file in os.listdir(self.bench_path_mhp):
                os.remove(os.path.join(self.bench_path_mhp), file)

    @staticmethod
    def _get_dev_num(filename: str):
        num_match = re.search(r'd_(\d+)', filename)
        if num_match:
            return int(num_match.group(1))
        else:
            return None

    @staticmethod
    def _get_mpi_proc_num(filename: str):
        num_match = re.search(r'n_(\d+)', filename)
        if num_match:
            return int(num_match.group(1))
        else:
            return None

    def shp_speedup(self):
        """
        This function plots the speedup gained
        by increasing the number of devices
        """
        time_results = dict()
        d_added = set()
        vec_size = 0
        reps = 0
        # naive approach, assuming that there is no timestamp
        for filename in os.listdir(self.bench_path_shp):
            d = Plotter._get_dev_num(filename)
            if d in d_added or d is None:
                continue
            d_added.add(d)
            with open(os.path.join(self.bench_path_shp, filename), 'r') as f:
                data = json.load(f)
                benchmarks = data["benchmarks"]
                copy = benchmarks[0]["real_time"]
                scale = benchmarks[1]["real_time"]
                add = benchmarks[2]["real_time"]
                triad = benchmarks[3]["real_time"]
                time_results[d] = BenchResults(copy, scale, add, triad)
                if vec_size != 0 and reps != 0:
                    continue
                vec_size = int(data["context"]["default_vector_size"])
                reps = int(data["context"]["default_repetitions"])

        time_results = dict(sorted(time_results.items()))
        copy, scale, add, triad = [1.0], [1.0], [1.0], [1.0]
        d_list = list(d_added)

        # no base value for 1 device
        if 1 not in d_list:
            return

        for i in range(1, len(time_results.keys())):
            keys = list(time_results.keys())
            d = keys[i]
            copy.append(time_results[1].copy / time_results[d].copy)
            scale.append(time_results[1].scale / time_results[d].scale)
            add.append(time_results[1].add / time_results[d].add)
            triad.append(time_results[1].triad / time_results[d].triad)

        plt.plot(d_list, copy, label='Stream_Copy/real_time')
        plt.plot(d_list, scale, label='Stream_Scale/real_time')
        plt.plot(d_list, add, label='Stream_Add/real_time')
        plt.plot(d_list, triad, label='Stream_Triad/real_time')

        plt.xticks(d_list)

        plt.xlabel('Devices')
        plt.ylabel('SHP Speedup')
        plt.title(f"Speedup for vec {vec_size} and {reps} reps")
        plt.legend()

        plt.savefig(f'{self.plots_path_shp}/shp_speedup.png', format='png')

    def _get_bench_vals(self, f):
        data = json.load(f)
        benchmarks = data["benchmarks"]
        copy = round(benchmarks[0]["bytes_per_second"] / 1e9, 2)
        scale = round(benchmarks[1]["bytes_per_second"] / 1e9, 2)
        add = round(benchmarks[2]["bytes_per_second"] / 1e9, 2)
        triad = round(benchmarks[3]["bytes_per_second"] / 1e9, 2)

        self.vec_size = int(data["context"]["default_vector_size"])
        self.reps = int(data["context"]["default_repetitions"])
        return BenchResults(copy, scale, add, triad)

    def load_bandwidth(self, mhp_shp: str = "shp"):
        self.shp_bandwidth_results = dict()
        self.mhp_bandwidth_results = dict()
        d_added = set()
        n_added = set()

        # load shp
        for filename in os.listdir(self.bench_path_shp):
            d = Plotter._get_dev_num(filename)
            if d in d_added or d is None or 'json' not in filename:
                continue
            d_added.add(d)
            with open(os.path.join(self.bench_path_shp, filename), 'r') as f:
                bench_result = self._get_bench_vals(f)
                self.shp_bandwidth_results[d] = bench_result

        self.shp_bandwidth_results = dict(
            sorted(self.shp_bandwidth_results.items())
        )
        self.d_list = list(d_added)

        # load mhp
        for filename in os.listdir(self.bench_path_mhp):
            n = Plotter._get_mpi_proc_num(filename)
            if n in n_added or n is None or 'json' not in filename:
                continue
            n_added.add(n)
            with open(os.path.join(self.bench_path_mhp, filename), 'r') as f:
                bench_result = self._get_bench_vals(f)
                self.mhp_bandwidth_results[n] = bench_result

        self.mhp_bandwidth_results = dict(
            sorted(self.mhp_bandwidth_results.items())
        )
        self.n_list = list(n_added)

    def _extract_bench_values(self, shp_mhp: str = "shp"):
        if shp_mhp == "shp":
            results = self.shp_bandwidth_results
        else:
            results = self.mhp_bandwidth_results

        copy, scale, add, triad = [], [], [], []
        for result in results.values():
            copy.append(result.copy)
            scale.append(result.scale)
            add.append(result.add)
            triad.append(result.triad)

        return copy, scale, add, triad

    def plot_bandwidth_shp(self):
        """
        Plots bandwidths for every number of devices from 1 to n (shp)
        """
        copy, scale, add, triad = self._extract_bench_values("shp")

        plt.plot(self.d_list, copy, label='Stream_Copy/bandwidth')
        plt.plot(self.d_list, scale, label='Stream_Scale/bandwidth')
        plt.plot(self.d_list, add, label='Stream_Add/bandwidth')
        plt.plot(self.d_list, triad, label='Stream_Triad/bandwidth')

        plt.xlabel('Devices')
        plt.ylabel('GBps')
        plt.title("Bandwidth")
        plt.legend()
        plt.savefig(f'{self.plots_path_shp}/shp_bandwidth.png', format='png')

    def plot_bandwidth_mhp(self):
        """
        Plots bandwidths for every number of mpi processes from 1 to n (mhp)
        """
        copy, scale, add, triad = self._extract_bench_values("mhp")

        plt.plot(self.n_list, copy, label='Stream_Copy/bandwidth')
        plt.plot(self.n_list, scale, label='Stream_Scale/bandwidth')
        plt.plot(self.n_list, add, label='Stream_Add/bandwidth')
        plt.plot(self.n_list, triad, label='Stream_Triad/bandwidth')

        plt.xlabel('MPI procs')
        plt.ylabel('GBps')
        plt.title("Bandwidth")
        plt.legend()
        plt.savefig(f'{self.plots_path_mhp}/mhp_bandwidth.png', format='png')


@click.command
@click.option('--shp', is_flag=True, help='Runs only shp benchmarks')
@click.option('--mhp', is_flag=True, help='Runs only mhp benchmarks')
def main(shp, mhp):
    plotter = Plotter()
    plotter.load_bandwidth()
    if shp:
        plotter.plot_bandwidth_shp()
    if mhp:
        plotter.plot_bandwidth_mhp()


if __name__ == '__main__':
    main()
