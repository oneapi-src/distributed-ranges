====================
 Distributed Ranges
====================

Main branch:

.. image:: https://github.com/intel-sandbox/personal.rscohn1.distributed-ranges/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/intel-sandbox/personal.rscohn1.distributed-ranges/actions/workflows/ci.yml

Proposal and reference implementation for Distributed Ranges.

The main branch is built on every commit and published at `latest
spec`_.

Create a python virtual environment and install dependencies::

  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

Activate virtual environment::

  source venv/bin/activate

Build the spec::

  make -C doc/spec html

Open in your browser: ``doc/spec/build/html/index.html``

Pre-commit testing::

  pre-commit run --all

Do pre-commit testing as part of commit::

  pre-commit install

.. _`latest spec`: https://stunning-fortnight-c2e7e025.pages.github.io/spec
