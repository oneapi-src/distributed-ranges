#!/bin/bash -e
# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

VDIR=regenerate-venv
python3 -m venv --clear ${VDIR}
source ${VDIR}/bin/activate
pip install -r base-requirements.txt
pip freeze > requirements.txt
reuse addheader --exclude-year --license BSD-3-Clause --copyright "Intel Corporation" requirements.txt
git diff requirements.txt
