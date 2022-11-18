====================
 Distributed Ranges
====================

.. image:: https://github.com/intel-sandbox/personal.rscohn1.distributed-ranges/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/intel-sandbox/personal.rscohn1.distributed-ranges/actions/workflows/ci.yml

Proposal and reference implementation for Distributed Ranges.

`Overview slides`_

.. _`Overview slides`: doc/Distributed%20Ranges.pdf

The documentation is built from main branch on every commit and
published at `latest spec`_ and `latest doxygen`_.

Environment Setup
=================

CPU requires g++ 10 or higher, mpi, and MKL. On Ubuntu 20.04::

  sudo apt install g++-10 openmpi

SYCL requires g++ 12 standard library. On Ubuntu 22.04::

  sudo apt install g++-12

SYCL requires a nightly build from the dpcpp open source project::

  wget https://github.com/intel/llvm/releases/download/sycl-nightly%2F20221029/dpcpp-compiler.tar.gz
  tar zxf dpcpp-compiler.tar.gz
  source dpcpp_cmpiler/startup.sh

If you want to build the document, you must install some python
packages. Create a python virtual environment and install
dependencies::

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

Doxygen html is at: ``doc/spec/build/doxygen-html/index.html``

Examples
========

Build and test examples with gcc on CPU::

  CXX=g++-10 cmake -B build
  make -C build -j all test

Enable SYCL examples::

  CXX=clang++ cmake -B build -DENABLE_SYCL=ON

See how example is run and the output::

  cd build
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

``pre-commit`` will automatically fix most issues. Do a ``git add`` to
add the changes and run ``pre-commit`` or ``git commit`` again.

.. _`latest spec`: https://stunning-fortnight-c2e7e025.pages.github.io/spec
.. _`latest doxygen`: https://stunning-fortnight-c2e7e025.pages.github.io/doxygen
