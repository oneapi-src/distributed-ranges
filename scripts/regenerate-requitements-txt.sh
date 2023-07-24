#!/bin/bash -e

VDIR=$HOME/tmp/drvenvforreqtxt
python3 -m venv $VDIR
. $VDIR/bin/activate
pip install -r base-requirements.txt
cat > requirements.txt << EOL
# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

EOL
pip freeze >> requirements.txt
git diff requirements.txt
