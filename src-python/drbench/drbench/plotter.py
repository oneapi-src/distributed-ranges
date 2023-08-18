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
    speedup_title = "Speedup vs DPL"

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
            cpu_sockets = (
                ranks if runtime == "SYCL" else ranks / cores_per_socket
            )
            for b in benchs:
                bname = b["name"].partition("/")[0]
                bname, btarget = Plotter.__name_target(bname, target, device)
                rtime = b["real_time"]
                bw = b["bytes_per_second"]
                rows.append(
                    {
                        "Benchmark": bname,
                        "Target": btarget,
                        "Ranks": ranks,
                        "Scaling": "weak" if weak_scaling else "strong",
                        Plotter.tbs_title: bw / 1e12,
                        Plotter.gbs_title: bw / 1e9,
                        "model": model,
                        "runtime": runtime,
                        "device": device,
                        "vsize": vsize,
                        "Number of GPU Tiles": ranks,
                        "Number of CPU Cores": cpu_cores,
                        "Number of CPU Sockets": cpu_sockets,
                        "rtime": rtime,
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
        plt.savefig(f"{file_name}.png")
        plt.savefig(f"{file_name}.pdf")

    stream_info = {
        "GPU": {
            "y_domain": [1, 2, 4, 8, 16],
            "y_title": tbs_title,
        },
        "CPU": {
            "y_domain": [100, 200, 400, 800],
            "y_title": gbs_title,
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
            "targets": ["MHP_SYCL_GPU", "SHP_SYCL_GPU"],
        },
        "CPU": {
            "x_title": "Number of CPU Sockets",
            "targets": ["MHP_SYCL_CPU", "MHP_DIRECT_CPU"],
        },
    }

    def __x_domain(self, db, target, x_title):
        points = db.loc[db["Target"] == target]
        val = points[x_title].values[0]
        last = points[x_title].values[-1]
        x_domain = []
        while val <= last:
            x_domain.append(val)
            val = 2 * val
        return x_domain

    def __bw_plot(self, benchmark, device, scaling):
        di = self.device_info[device]
        x_title = di["x_title"]
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

        def points(target):
            p = db.loc[db["Target"] == target]
            return [p[x_title].values, p[y_title].values, target]

        self.__plot(
            benchmark,
            x_title,
            y_title,
            self.__x_domain(db, di["targets"][0], x_title),
            y_domain,
            [points(target) for target in di["targets"]],
            fname,
        )

    def __speedup_plot(self, benchmark, device, scaling):
        di = self.device_info[device]
        x_title = self.device_info[device]["x_title"]
        y_title = self.speedup_title

        db = self.db.copy()
        db = db.loc[db["Benchmark"] == benchmark]
        db = db.loc[db["Scaling"] == scaling]
        db = db.loc[db["device"] == device]
        db = db.sort_values(by=["Benchmark", "Target", x_title])

        x_domain = self.__x_domain(db, di["targets"][0], x_title)
        fname = f"{self.prefix}-{benchmark}-{device}-{scaling}"
        click.echo(f"writing {fname}")
        db.to_csv(f"{fname}.csv")

        dpl = db.loc[db["Target"] == f"DPL_{device}"]
        dpl = dpl.loc[dpl["Ranks"] == 1]
        dpl_rtime = dpl["rtime"].values[0]

        def points(target):
            p = db.loc[db["Target"] == target]
            return [p[x_title].values, dpl_rtime / p["rtime"].values, target]

        self.__plot(
            benchmark,
            x_title,
            y_title,
            x_domain,
            x_domain,
            [points(target) for target in di["targets"]],
            fname,
            # display_perfect_scaling=False
        )

    def create_plots(self):
        for device in ["CPU", "GPU"]:
            for bench in [
                "Stream_Copy",
                "Stream_Scale",
                "Stream_Add",
                "Stream_Triad",
            ]:
                self.__bw_plot(bench, device, "strong")
            for bench in [
                "Inclusive_Scan",
                "Reduce",
            ]:
                for scaling in ["strong"]:
                    self.__speedup_plot(bench, device, scaling)
