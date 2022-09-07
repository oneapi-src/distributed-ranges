====================
 Distributed Ranges
====================

Proposal and reference implementation for Distributed Ranges.

Create a python virtual environment and install preqrequisites::

  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

Activate virtual environment::

  source venv/bin/activate

Build the spec::

  make -C doc/spec

Open in your browser: ``doc/spec/build/html/index.html``

Pre-commit testing::

  pre-commit

Do pre-commit testing as part of commit::

  pre-commit install
