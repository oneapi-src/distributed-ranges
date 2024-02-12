# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime
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
    speedup_title = "Speedup"
    gpus_num_title = "Number of GPU Tiles"
    sockets_num_title = "Number of CPU Sockets"

    device_info = {
        "GPU": {
            "x_title": gpus_num_title,
            "targets": ["MHP_SYCL_GPU", "SHP_SYCL_GPU"],
        },
        "CPU": {
            "x_title": sockets_num_title,
            "targets": ["MHP_SYCL_CPU", "MHP_DIRECT_CPU"],
        },
    }

    @staticmethod
    def __name_target(bname, target, device):
        names = bname.split("_")
        last = names[-1]
        if last == "Reference":
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
                device_memory = ctx["device-memory"] == "1"
            except KeyError:
                click.fail(f"could not parse context of {fname}")
            benches = fdata["benchmarks"]
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
            for b in benches:
                bname = b["name"].partition("/")[0]
                if bname in ["DRSortFixture", "SyclSortFixture"]:
                    bname = b["name"].split("/")[1]
                    print("map", bname)
                benchmark, btarget = Plotter.__name_target(
                    bname, target, device
                )
                rtime = b["real_time"]
                bw = b["bytes_per_second"] if "bytes_per_second" in b else 1
                rows.append(
                    {
                        "bench_name": bname,
                        "Benchmark": benchmark,
                        "Target": btarget,
                        "Ranks": ranks,
                        "Scaling": "weak" if weak_scaling else "strong",
                        "Device Memory": device_memory,
                        Plotter.tbs_title: bw / 1e12,
                        Plotter.gbs_title: bw / 1e9,
                        "model": model,
                        "runtime": runtime,
                        "device": device,
                        "vsize": vsize,
                        Plotter.gpus_num_title: ranks,
                        "Number of CPU Cores": cpu_cores,
                        Plotter.sockets_num_title: cpu_sockets,
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

        plt.legend(loc="best", fontsize=10)

        plt.figtext(0.01, 0.01, datetime.datetime.now().strftime("%x %X"))

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

    @staticmethod
    def __x_domain(db, target, x_title):
        points = db.loc[db["Target"] == target]
        val = points[x_title].values[0]
        last = points[x_title].values[-1]
        x_domain = [val]
        while True:
            x_domain.append(val)
            if val >= last:
                break
            val = 2 * val
        return x_domain

    def __find_targets(self, db, device):
        targets = []
        for target in self.device_info[device]["targets"]:
            tdb = db.loc[db["Target"] == target]
            if tdb.shape[0] == 0:
                click.echo(f"  no data for {target}")
            else:
                targets.append(target)
        return targets

    def __bw_plot(self, benchmark, device):
        scaling = "strong"
        x_title = self.device_info[device]["x_title"]
        bi = self.benchmark_info[benchmark][device]
        y_domain = bi["y_domain"]
        y_title = bi["y_title"]

        db = self.db.copy()
        db = db.loc[
            (db["Benchmark"] == benchmark)
            & (db["Scaling"] == scaling)
            & (db["device"] == device)
        ]
        db = db.sort_values(by=["Benchmark", "Target", x_title])

        targets = self.__find_targets(db, device)

        if db.shape[0] == 0 or len(targets) == 0:
            click.echo(f"no data for {benchmark} {device} {scaling}")
            return

        fname = f"{self.prefix}-{benchmark}-{device}"
        click.echo(f"writing {fname}")
        db.to_csv(f"{fname}.csv")

        def points(target):
            p = db.loc[db["Target"] == target]
            return [p[x_title].values, p[y_title].values, target]

        self.__plot(
            benchmark,
            x_title,
            y_title,
            self.__x_domain(db, targets[0], x_title),
            y_domain,
            [points(target) for target in targets],
            fname,
        )

    def __speedup_plot(
        self, benchmark, device, benchmark_name=None, reference_name=None
    ):
        fname = f"{self.prefix}-{benchmark}-{device}"
        click.echo(f"writing {fname}")

        if not benchmark_name:
            benchmark_name = f"{benchmark}_DR"
        if not reference_name:
            reference_name = f"{benchmark}_Reference"

        x_title = self.device_info[device]["x_title"]
        y_title = self.speedup_title

        db = self.db.copy()
        db = db.loc[(db["Benchmark"] == benchmark) & (db["device"] == device)]
        db = db.sort_values(by=["Target", x_title])

        targets = self.__find_targets(db, device)

        if db.shape[0] == 0 or len(targets) == 0:
            click.echo(f"  no data for {benchmark} {device}")
            return

        xy_domain = self.__x_domain(db, targets[0], x_title)
        db.to_csv(f"{fname}.csv")

        reference = db.loc[
            (db["device"] == device)
            & (db["bench_name"] == reference_name)
            & (db["Ranks"] == 1)
        ]
        if reference.shape[0] == 0:
            click.echo(f"  no reference data for {benchmark} {device}")
            return
        reference_rtime = reference["rtime"].values[0]

        lines = []
        for scaling in ["weak", "strong"]:
            for target in targets:
                for device_memory in [True, False]:
                    p = db.loc[
                        (db["bench_name"] == benchmark_name)
                        & (db["Target"] == target)
                        & (db["Scaling"] == scaling)
                        & (db["Device Memory"] == device_memory)
                    ]
                    if p.shape[0] == 0:
                        click.echo(
                            f"  no data for {benchmark_name} {target}"
                            f"{scaling} device_memory:{device_memory}"
                        )
                        continue

                    total_time = p["rtime"].values / (
                        1 if scaling == "strong" else p["Ranks"].values
                    )
                    label = target
                    if scaling == "weak":
                        label += " weak scaling"
                    if device_memory:
                        label += " device memory"
                    lines.append(
                        [
                            p[x_title].values,
                            reference_rtime / total_time,
                            label,
                        ]
                    )

        self.__plot(
            benchmark,
            x_title,
            y_title,
            xy_domain,
            xy_domain,
            lines,
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
                self.__bw_plot(bench, device)
            for bench in [
                "BlackScholes",
                "DotProduct",
                "Gemm",
                "Exclusive_Scan",
                "Inclusive_Scan",
                "Reduce",
                # "Sort",
                "Stencil2D",
            ]:
                self.__speedup_plot(bench, device)
            # Use the DR version as the reference
            for bench in [
                # Talk to Jeongnim. Can MKL fft do 3d without
                # separate transpose?
                "FFT3D",
                # Talk to Tuomas.
                "WaveEquation",
            ]:
                self.__speedup_plot(
                    bench, device, reference_name=f"{bench}_DR"
                )
