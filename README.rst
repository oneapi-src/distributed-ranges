.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

====================
 Distributed Ranges
====================

.. image:: https://github.com/oneapi-src/distributed-ranges/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/oneapi-src/distributed-ranges/actions/workflows/ci.yml

Productivity library for distributed and partitioned memory based on
C++ Ranges.

`Overview slides`_

.. _`Overview slides`: doc/presentations/Distributed%20Ranges.pdf

The documentation is built from main branch on every commit and
published at `latest spec`_ and `latest doxygen`_.

Environment Setup
=================

CPU & SYCL (MPI) requires g++ 10 or higher, mpi, and MKL. On Ubuntu
20.04::

  sudo apt install g++-10 libopenmpi-dev

SYCL (SHP) requires g++ 12 standard library. On Ubuntu 22.04::

  sudo apt install g++-12

SYCL (SHP) requires a nightly build from the dpcpp open source project. If
you are targeting intel gpu::

  wget https://github.com/intel/llvm/releases/download/sycl-nightly%2F20221029/dpcpp-compiler.tar.gz
  tar zxf dpcpp-compiler.tar.gz
  source dpcpp_compiler/startup.sh

If you are targeting cuda::

  git clone https://github.com/intel/llvm
  cd llvm
  git checkout sycl-nightly/20221029
  python buildbot/configure.py --cuda

Copy `startup.sh` from the open source binary build, or create it at
`build/install/startup.sh`::

    export SYCL_BUNDLE_ROOT=$(realpath $(dirname "${BASH_SOURCE[0]}"))
    export PATH=$SYCL_BUNDLE_ROOT/bin:$PATH
    export CPATH=$SYCL_BUNDLE_ROOT/include:$CPATH
    export LIBRARY_PATH=$SYCL_BUNDLE_ROOT/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$SYCL_BUNDLE_ROOT/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$SYCL_BUNDLE_ROOT/linux/lib/x64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$SYCL_BUNDLE_ROOT/lib/oclgpu:$LD_LIBRARY_PATH

Then::

  source build/install/startup.sh

If you want to build the document or run the pre-commit checks, you
must install some python packages. Create a python virtual environment
and install dependencies::

  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

Activate virtual environment::

  source venv/bin/activate

Examples
========

Build and test examples with gcc on CPU::

  CXX=g++-10 cmake -B build
  make -C build -j all test

Enable SYCL examples::

  CXX=clang++ cmake -B build -DENABLE_SYCL=ON

Enable SYCL-MPI examples::

  CXX=clang++ cmake -B build -DENABLE_SYCL_MPI=ON

See how example is run and the output::

  cd build
  ctest -VV

Logging
=======

Add this to your main to enable logging::

  std::ofstream logfile(fmt::format("dr.{}.log", comm_rank));
  lib::drlog.set_file(logfile);

Adding DR to a project
======================

If your project uses CMAKE, add the following to your
``CMakeLists.txt`` to download the library::

  find_package(MPI REQUIRED)
  include(FetchContent)
  FetchContent_Declare(
    dr
    GIT_REPOSITORY https://github.com/oneapi-src/distributed-ranges.git
    GIT_TAG main
    )
  FetchContent_MakeAvailable(dr)

The above will define targets that can be included in your project::

  target_link_libraries(<application> MPI::MPI_CXX DR::mpi)

If your project does not use CMAKE, then you need to download the
library, and install it into a prefix::

  git clone https://github.com/oneapi-src/distributed-ranges.git dr
  cd dr
  cmake -B build -DCMAKE_INSTALL_PREFIX=<prefix>
  make -C build install
  cmake -B build-fmt -DCMAKE_INSTALL_PREFIX=<prefix> build/_deps/cpp-format-src
  make -C build-fmt install

Use ``-I`` and ``-L`` to find headers and libs during compilation::

  g++ -std=c=++20 -I <prefix>/include -L <prefix>/lib -L /opt/intel/oneapi/mpi/latest/lib/release -lfmt -lmpicxx -lmpi


Developer Information
=====================

Submitting a PR
---------------

Follow the standard github workflow. Fork this repo, clone your fork,
make changes, commit to a new branch, push branch to your fork. Submit
a PR from your fork.

The CI runs static checks and runs the test system. See `pre-commit`_
for information on failing static checks.

See also
--------

`Fuzz Testing`_
  Fuzz testing of distributed ranges APIs

`Spec Editing`_
  Editing the API document

`Print Type`_
  Print types at compile time:

`Testing`_
  Test system maintenance

.. _`Testing`: doc/developer/testing
.. _`pre-commit`: doc/developer/testing/pre-commit.rst
.. _`Spec Editing`: doc/spec/README.rst
.. _`Fuzz Testing`: test/fuzz/README.rst
.. _`Print Type`: https://stackoverflow.com/a/14617848/2525421
.. _`latest spec`: https://oneapi-src.github.io/distributed-ranges/spec
.. _`latest doxygen`: https://oneapi-src.github.io/distributed-ranges/doxygen
