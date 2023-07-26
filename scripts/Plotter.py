#!/usr/bin/env python3

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import re
import shutil

import click
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self) -> None:
        self.plots_path_shp = "scripts"
        self.plots_path_mhp = "scripts"
        self.bench_path_shp = "scripts"
        self.bench_path_mhp = "scripts"
        self.create_directories()

    def create_directories(self):
        os.makedirs(self.plots_path_shp, exist_ok=True)
        os.makedirs(self.plots_path_mhp, exist_ok=True)
        os.makedirs(self.bench_path_shp, exist_ok=True)
        os.makedirs(self.bench_path_mhp, exist_ok=True)

    def clean_directories(self):
        if os.path.exists(self.plots_path_shp) and os.path.isdir(
            self.plots_path_shp
        ):
            shutil.rmtree(self.plots_path_shp)
        if os.path.exists(self.plots_path_mhp) and os.path.isdir(
            self.plots_path_mhp
        ):
            shutil.rmtree(self.plots_path_mhp)
        if os.path.exists(self.bench_path_shp) and os.path.isdir(
            self.bench_path_shp
        ):
            shutil.rmtree(self.bench_path_shp)
        if os.path.exists(self.bench_path_mhp) and os.path.isdir(
            self.bench_path_mhp
        ):
            shutil.rmtree(self.bench_path_mhp)

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
        # naive approach, assuming that there is no timestamp
        for filename in os.listdir(self.bench_path_shp):
            d = Plotter._get_dev_num(filename)
            if d in d_added or d is None:
                continue
            d_added.add(d)
            with open(os.path.join(self.bench_path_shp, filename), 'r') as f:
                data = json.load(f)
                benchmarks = data["benchmarks"]
                benchmarks_values = []
                benchmarks_names = []
                for i in range(len(benchmarks)):
                    benchmarks_values.append(benchmarks[i]["real_time"])
                    benchmarks_names.append(benchmarks[i]["name"])

                time_results[d] = benchmarks_values

                self.vec_size = int(data["context"]["default_vector_size"])
                self.reps = int(data["context"]["default_repetitions"])

        time_results = dict(sorted(time_results.items()))
        bench_results = [[1.0] for _ in range(len(benchmarks_names))]

        d_list = list(d_added)
        # no base value for 1 device
        if 1 not in d_list:
            return

        for i in range(1, len(time_results.keys())):
            keys = list(time_results.keys())
            d = keys[i]
            for j in range(len(bench_results)):
                bench_results[j].append(
                    time_results[1][j] / time_results[d][j]
                )

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        for i, bench_result in enumerate(bench_results):
            ax.plot(d_list, bench_result, label=f'{benchmarks_names[i]}')

        ax.set_xticks(d_list)
        ax.set_xlabel('Devices')
        ax.set_ylabel('SHP Speedup')
        plt.title(f"Speedup for vec {self.vec_size} and {self.reps} reps")
        plt.legend()
        plt.savefig(
            f'{self.plots_path_shp}/shp_speedup.png',
            format='png',
            dpi=200,
            bbox_inches='tight',
        )
        plt.close(fig)

    def _get_bench_vals(self, f):
        data = json.load(f)
        benchmarks = data["benchmarks"]
        benchmarks_rounded = []
        benchmarks_names = []
        for i in range(len(benchmarks)):
            benchmarks_rounded.append(
                round(benchmarks[i]["bytes_per_second"] / 1e9, 2)
            )
            benchmarks_names.append(benchmarks[i]["name"])

        self.vec_size = int(data["context"]["default_vector_size"])
        self.reps = int(data["context"]["default_repetitions"])

        self.benchmark_names = benchmarks_names

        return benchmarks_rounded

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

        results_by_bench = [[] for _ in range(len(list(results.values())[0]))]
        for result in results.values():
            for i in range(len(result)):
                results_by_bench[i].append(result[i])
        return results_by_bench

    def plot_bandwidth_shp(self):
        """
        Plots bandwidths for every number of devices from 1 to n (shp)
        """
        benchmarks = self._extract_bench_values("shp")
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        for i, benchmark in enumerate(benchmarks):
            ax.plot(
                self.d_list,
                benchmark,
                label=f'{self.benchmark_names[i]}/bandwidth',
            )
        ax.set_xticks(self.d_list)
        ax.set_xlabel('Devices')
        ax.set_ylabel('GBps')
        plt.title("Bandwidth")
        plt.legend()
        plt.savefig(
            f'{self.plots_path_shp}/shp_bandwidth.png',
            format='png',
            dpi=200,
            bbox_inches='tight',
        )
        plt.close(fig)

    def plot_bandwidth_mhp(self):
        """
        Plots bandwidths for every number of mpi processes from 1 to n (mhp)
        """
        benchmarks = self._extract_bench_values("mhp")
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        for i, benchmark in enumerate(benchmarks):
            ax.plot(
                self.d_list,
                benchmark,
                label=f'{self.benchmark_names[i]}/bandwidth',
            )
        ax.set_xticks(self.d_list)
        ax.set_xlabel('MPI procs')
        ax.set_ylabel('GBps')
        plt.title("Bandwidth")
        plt.legend()
        plt.savefig(
            f'{self.plots_path_mhp}/mhp_bandwidth.png',
            format='png',
            dpi=200,
            bbox_inches='tight',
        )
        plt.close(fig)


@click.command
@click.option('--shp', is_flag=True, help='Runs only shp benchmarks')
@click.option('--mhp', is_flag=True, help='Runs only mhp benchmarks')
def main(shp, mhp):
    plotter = Plotter()
    plotter.load_bandwidth()
    if shp:
        plotter.plot_bandwidth_shp()
        plotter.shp_speedup()
    if mhp:
        plotter.plot_bandwidth_mhp()


if __name__ == '__main__':
    main()
