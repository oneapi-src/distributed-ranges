# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


class Config:
    analysis_id = ''


def analysis_file_prefix(analysis_id: str):
    return f'dr-bench-{analysis_id}'
