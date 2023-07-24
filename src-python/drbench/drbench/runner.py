# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# TODO: more stuff is going to be moved here from drbench.py

import subprocess

import click


def execute(command: str, ctx):
    click.echo(command)
    if not ctx.obj['DRY_RUN']:
        subprocess.run(command, shell=True, check=True)
