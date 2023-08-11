# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import re

import click
import pandas as pd
import seaborn as sns


class Plotter:
    bandwidth_title = "Memory Bandwidth (TBps)"

    @staticmethod
    def __name_target(bname, target, device):
        names = bname.split("_")
        last = names[-1]
        if last in ["DPL", "Std", "Serial"]:
            bname = "_".join(names[0:-1])
            target = f"{last}_{device}"
        elif last == "DR":
            bname = "_".join(names[0:-1])
        return bname, target

    @staticmethod
    def __import_file(fname: str, rows):
        with open(fname) as f:
            fdata = json.load(f)
            ctx = fdata["context"]
            try:
                vsize = int(ctx["default_vector_size"])
                ranks = int(ctx["ranks"])
                target = ctx["target"]
                model = ctx["model"]
                runtime = ctx["runtime"]
                device = ctx["device"]
                weak_scaling = ctx["weak-scaling"]
            except KeyError:
                print(f"could not parse context of {fname}")
                raise
            benchs = fdata["benchmarks"]
            cores_per_socket = int(
                re.search(
                    r"Core\(s\) per socket:\s*(\d+)", ctx["lscpu"]
                ).group(1)
            )
            cpu_cores = (
                ranks if runtime == "DIRECT" else ranks * cores_per_socket
            )
            for b in benchs:
                bname = b["name"].partition("/")[0]
                bname, btarget = Plotter.__name_target(bname, target, device)
                rtime = b["real_time"]
                bw = b["bytes_per_second"]
                rows.append(
                    {
                        "Target": btarget,
                        "model": model,
                        "runtime": runtime,
                        "device": device,
                        "vsize": vsize,
                        "Benchmark": bname,
                        "Ranks": ranks,
                        "GPU Tiles": ranks,
                        "CPU Cores": cpu_cores,
                        "rtime": rtime,
                        "Weak Scaling": weak_scaling,
                        Plotter.bandwidth_title: bw / 1e12,
                    }
                )

    # db is created which looks something like this:
    #           mode  vsize          benchmark  nprocs      rtime            bw
    # 0   MHP_NOSYCL  20000   Stream_Copy       1   0.234987  1.361779e+11
    # 1   MHP_NOSYCL  20000  Stream_Scale       1   0.240879  1.328468e+11
    # 2   MHP_NOSYCL  20000    Stream_Add       1   0.329298  1.457645e+11
    # ..         ...    ...           ...     ...        ...           ...
    # 62     MHP_GPU  40000    Stream_Add       4  21.716973  4.420506e+09
    # 63     MHP_GPU  40000  Stream_Triad       4  21.714421  4.421025e+09
    def __init__(self, prefix):
        rows = []
        for fname in glob.glob(f"{prefix}-*.json"):
            click.echo(f"found file {fname}")
            Plotter.__import_file(fname, rows)

        self.db = pd.DataFrame(rows)

        # helper structures that can be used to define plots
        self.vec_sizes = self.db["vsize"].unique()
        self.vec_sizes.sort()
        self.max_vec_size = self.vec_sizes[-1]
        self.db_maxvec = self.db.loc[(self.db["vsize"] == self.max_vec_size)]

        self.ranks = self.db["Ranks"].unique()
        self.ranks.sort()
        self.prefix = prefix

    def __make_plot(self, fname, data, **kwargs):
        plot = sns.relplot(data=data, kind="line", marker="d", **kwargs)
        plot.savefig(
            f"{self.prefix}-{fname}.png", dpi=200, bbox_inches="tight"
        )
        csv_name = f"{self.prefix}-{fname}.csv"
        click.echo(f"writing {csv_name}")
        sorted = data.sort_values(by=["Benchmark", "Target", "Ranks"])
        sorted.to_csv(csv_name)

    def __stream_scaling_plots(self, weak_scaling: bool = False):
        db = self.db_maxvec.loc[
            self.db["Benchmark"].str.startswith("Stream_")
        ].copy()

        if weak_scaling:
            db = db.loc[db["Weak Scaling"]]
            gpu_title = "stream_weak_scaling_gpu"
            cpu_title = "stream_weak_scaling_cpu"
        else:
            gpu_title = "stream_strong_scaling_gpu"
            cpu_title = "stream_strong_scaling_cpu"

        db_gpu = db.loc[db["device"] == "GPU"]

        self.__make_plot(
            gpu_title,
            db_gpu,
            x="GPU Tiles",
            y=Plotter.bandwidth_title,
            col="Benchmark",
            style="Target",
        )

        db_cpu = db.loc[db["device"] == "CPU"]

        self.__make_plot(
            cpu_title,
            db_cpu,
            x="CPU Cores",
            y=Plotter.bandwidth_title,
            col="Benchmark",
            style="Target",
        )

    def __algorithm_plots(self, weak_scaling: bool = False):
        m = self.db_maxvec

        db = m.loc[
            m["Benchmark"].isin(["Black_Scholes", "Inclusive_Scan", "Reduce"])
        ].copy()

        if weak_scaling:
            db = db.loc[db["Weak Scaling"]]
            gpu_title = "algorithms_gpu_weak_scaling"
            cpu_title = "algorithms_cpu_weak_scaling"
        else:
            gpu_title = "algorithms_gpu_strong_scaling"
            cpu_title = "algorithms_cpu_strong_scaling"

        db_gpu = db.loc[db["device"] == "GPU"]

        self.__make_plot(
            gpu_title,
            db_gpu,
            x="GPU Tiles",
            y=Plotter.bandwidth_title,
            col="Benchmark",
            style="Target",
        )

        db_cpu = db.loc[db["device"] == "CPU"]

        self.__make_plot(
            cpu_title,
            db_cpu,
            x="CPU Cores",
            y=Plotter.bandwidth_title,
            col="Benchmark",
            style="Target",
        )

    def create_plots(self):
        sns.set_theme(style="ticks")

        self.__stream_scaling_plots(weak_scaling=False)
        self.__algorithm_plots(weak_scaling=False)

        self.__stream_scaling_plots(weak_scaling=True)
        self.__algorithm_plots(weak_scaling=True)
