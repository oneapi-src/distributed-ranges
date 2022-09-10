====================
 Distributed Ranges
====================

.. image:: https://github.com/intel-sandbox/personal.rscohn1.distributed-ranges/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/intel-sandbox/personal.rscohn1.distributed-ranges/actions/workflows/ci.yml

Proposal and reference implementation for Distributed Ranges.

The documentation is built from main branch on every commit and
published at `latest spec`_ and `latest doxygen`_.

Environment Setup
=================

Create a python virtual environment and install dependencies::

  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

Activate virtual environment::

  source venv/bin/activate

Editing the Spec
================

Build the spec::

  make -C doc/spec html

Open in your browser: ``doc/spec/build/html/index.html``

Examples
========

Build and test examples::

  mkdir build
  cd build
  cmake ..
  make -j all test

See how example is run and the output::

  ctest -VV

Contributing
============

Follow the standard github workflow. Fork this repo, clone your fork,
make changes, commit to a new branch, push branch to your fork. Submit
a PR from your fork.

The CI runs some formatting checks. It is easiest to resolve errors
with your local build before submitting the PR.

Run pre-commit checks::

  pre-commit run --all

Do pre-commit testing as part of commit::

  pre-commit install

.. _`latest spec`: https://stunning-fortnight-c2e7e025.pages.github.io/spec
.. _`latest doxygen`: https://stunning-fortnight-c2e7e025.pages.github.io/doxygen
