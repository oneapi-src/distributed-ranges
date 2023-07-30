# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
from collections import namedtuple

import click
import pandas as pd
import seaborn as sns
from drbench import common

# only common_config for now, add plotting options here if needed
PlottingConfig = namedtuple(
    "PlottingConfig",
    "common_config",
)


class Plotter:
    @staticmethod
    def __is_our_file(fname: str, analysis_id: str):
        files_prefix = common.analysis_file_prefix(analysis_id)
        if not fname.startswith(files_prefix):
            return False
        if fname.startswith(files_prefix + ".rank000"):
            return True
        if fname.startswith(files_prefix + ".rank"):
            return False
        return True

    @staticmethod
    def __import_file(fname: str, rows):
        with open(fname) as f:
            fdata = json.load(f)
            ctx = fdata["context"]
            vsize = int(ctx["default_vector_size"])
            ranks = int(ctx["ranks"])
            model = ctx["model"]
            device = ctx["device"]
            runtime = ctx["runtime"]
            benchs = fdata["benchmarks"]
            config = f"{model}-{device}-{runtime}"
            for b in benchs:
                bname = b["name"].partition("/")[0]
                rtime = b["real_time"]
                bw = b["bytes_per_second"]
                rows.append(
                    {
                        "Config": config,
                        "vsize": vsize,
                        "Benchmark": bname,
                        "Ranks": ranks,
                        "rtime": rtime,
                        "Memory Bandwidth": bw,
                        "Model": model,
                        "Device": device,
                        "Runtime": runtime,
                    }
                )

    # db is created which looks something like this:
    #          Config  vsize     Benchmark   Ranks      rtime            bw
    # 0   MHP_NOSYCL  20000   Stream_Copy       1   0.234987  1.361779e+11
    # 1   MHP_NOSYCL  20000  Stream_Scale       1   0.240879  1.328468e+11
    # 2   MHP_NOSYCL  20000    Stream_Add       1   0.329298  1.457645e+11
    # ..         ...    ...           ...     ...        ...           ...
    # 62     MHP_GPU  40000    Stream_Add       4  21.716973  4.420506e+09
    # 63     MHP_GPU  40000  Stream_Triad       4  21.714421  4.421025e+09
    def __init__(self, plotting_config: PlottingConfig):
        rows = []
        for fname in glob.glob("*json"):
            if Plotter.__is_our_file(
                fname, plotting_config.common_config.analysis_id
            ):
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

        self.Configs = self.db["Config"].unique()

    @staticmethod
    def __make_plot(fname, data, **kwargs):
        plot = sns.relplot(data=data, kind="line", **kwargs)
        plot.savefig(f"{fname}.png")

    def __stream_bandwidth_plots(self, device):
        Plotter.__make_plot(
            f"stream_bw-{device}",
            self.db_maxvec.loc[self.db["Benchmark"].str.startswith("Stream_")],
            x="Ranks",
            y="Memory Bandwidth",
            col="Config",
            hue="Benchmark",
        )

    def __stream_strong_scaling_plots(self, device):
        db = self.db_maxvec.loc[
            self.db["Benchmark"].str.startswith("Stream_")
        ].copy()

        ref_stream = sorted(db["Benchmark"].unique())[0]
        ref_config = sorted(db["Config"].unique())[0]
        ref_ranks = sorted(db["Ranks"].unique())[0]
        # take value of reference stream/config/rank - can it be easier taken?
        scale_factor = (
            db.loc[
                (db["Config"] == ref_config)
                & (db["Benchmark"] == ref_stream)
                & (db["Ranks"] == ref_ranks)
            ]
            .squeeze()
            .at["Memory Bandwidth"]
        )

        click.echo(
            f"stream strong scalling scalled by {ref_stream} {ref_config}"
            f" ranks:{ref_ranks} eq {scale_factor}"
        )
        db["Memory Bandwidth"] /= scale_factor
        db["Normalized Memory Bandwidth"] = (
            db["Memory Bandwidth"] / scale_factor
        )

        Plotter.__make_plot(
            f"stream_strong_scaling-{device}",
            db,
            x="Ranks",
            y="Normalized Memory Bandwidth",
            col="Benchmark",
            hue="Config",
        )

    def create_plots(self):
        db = self.db_maxvec.loc[
            self.db["Benchmark"].str.startswith("Stream_")
        ].copy()
        device = sorted(db["Device"].unique())[0]
        sns.set_theme(style="ticks")

        self.__stream_bandwidth_plots(device)
        self.__stream_strong_scaling_plots(device)
