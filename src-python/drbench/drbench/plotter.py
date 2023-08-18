# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import re

import click
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


class Plotter:
    tbs_title = "Bandwidth (TB/s)"
    gbs_title = "Bandwidth (GB/s)"

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
                weak_scaling = ctx["weak-scaling"] == "1"
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
                        "Number of GPU Tiles": ranks,
                        "Number of CPU Cores": cpu_cores,
                        "rtime": rtime,
                        "Scaling": "weak" if weak_scaling else "strong",
                        Plotter.tbs_title: bw / 1e12,
                        Plotter.gbs_title: bw / 1e9,
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
        self.ranks = self.db["Ranks"].unique()
        self.ranks.sort()
        self.prefix = prefix

    def __plot(
        self,
        title,
        x_title,
        y_title,
        x_domain,
        y_domain,
        datasets,
        file_name,
        flat_lines=[],
        display_perfect_scaling=True,
        offset_pct=0.15,
    ):
        fig, ax = plt.subplots()

        if display_perfect_scaling:
            perfect_scaling_start = datasets[0][1][0]
            perfect_scaling = [
                (perfect_scaling_start - offset_pct * perfect_scaling_start)
                * x
                / x_domain[0]
                for x in x_domain
            ]
            ax.loglog(
                x_domain,
                perfect_scaling,
                label="perfect scaling",
                linestyle="dashed",
                color="grey",
            )

        for flat_line in flat_lines:
            single_point = flat_line[0]
            label = flat_line[1]
            color = flat_line[2]
            ax.axhline(single_point, label=label, color=color)

        markers = ["s", ".", "^", "o"]
        for marker, dataset in zip(markers, datasets):
            domain = dataset[0]
            y_points = dataset[1]
            label = dataset[2]
            ax.loglog(
                domain,
                y_points,
                label=label,
                marker=marker,
                markerfacecolor="white",
            )

        ax.minorticks_off()
        ax.set_xticks(x_domain)
        ax.set_xticklabels([str(x) for x in x_domain])

        yticks = y_domain
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

        ax.set_yticks(yticks)

        plt.title(title, fontsize=18)
        plt.xlabel(x_title, fontsize=12)
        plt.ylabel(y_title, fontsize=12)
        plt.rcParams.update({"font.size": 12})

        plt.legend(loc="best")

        plt.tight_layout()
        plt.savefig(file_name)

    stream_info = {
        "GPU": {
            "y_domain": [1, 2, 4, 8, 16],
            "y_title": tbs_title,
            "targets": ["MHP_SYCL_GPU", "SHP_SYCL_GPU"],
        },
        "CPU": {
            "y_domain": [100, 200, 400, 800],
            "y_title": gbs_title,
            "targets": ["MHP_SYCL_CPU", "MHP_DIRECT_CPU"],
        },
    }

    benchmark_info = {
        "Stream_Copy": stream_info,
        "Stream_Add": stream_info,
        "Stream_Scale": stream_info,
        "Stream_Triad": stream_info,
    }

    device_info = {
        "GPU": {
            "x_title": "Number of GPU Tiles",
        },
        "CPU": {
            "x_title": "Number of CPU Cores",
        },
    }

    def __line_info(self, db, target, x_title, y_title):
        points = db.loc[db["Target"] == target]
        return [points[x_title].values, points[y_title].values, target]

    def __x_domain(self, db, target, x_title):
        points = db.loc[db["Target"] == target]
        val = points[x_title].values[0]
        last = points[x_title].values[-1]
        x_domain = []
        while val <= last:
            x_domain.append(val)
            val = 2 * val
        return x_domain

    def __bench_plot(self, benchmark, device, scaling):
        x_title = self.device_info[device]["x_title"]
        bi = self.benchmark_info[benchmark][device]
        y_domain = bi["y_domain"]
        y_title = bi["y_title"]

        db = self.db.copy()
        db = db.loc[db["Benchmark"] == benchmark]
        db = db.loc[db["Scaling"] == scaling]
        db = db.loc[db["device"] == device]
        db = db.sort_values(by=["Benchmark", "Target", x_title])

        fname = f"{self.prefix}-{benchmark}-{device}-{scaling}"
        click.echo(f"writing {fname}")
        db.to_csv(f"{fname}.csv")

        self.__plot(
            benchmark,
            x_title,
            y_title,
            self.__x_domain(db, bi["targets"][0], x_title),
            y_domain,
            [
                self.__line_info(db, target, x_title, y_title)
                for target in bi["targets"]
            ],
            f"{fname}.png",
        )

    def create_plots(self):
        for device in ["CPU", "GPU"]:
            for bench in [
                "Stream_Copy",
                "Stream_Scale",
                "Stream_Add",
                "Stream_Triad",
            ]:
                self.__bench_plot(bench, device, "strong")
