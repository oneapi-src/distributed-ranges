#!/bin/bash -e

VDIR=$HOME/tmp/drvenvforreqtxt
python3 -m venv --clear $VDIR
. $VDIR/bin/activate
pip install -r base-requirements.txt
pip freeze > requirements.txt
reuse addheader --exclude-year --license BSD-3-Clause --copyright "Intel Corporation" requirements.txt
git diff requirements.txt
