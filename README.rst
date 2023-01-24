.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

====================
 Distributed Ranges
====================

.. image:: https://github.com/oneapi-src/distributed-ranges/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/oneapi-src/distributed-ranges/actions/workflows/ci.yml

Proposal and reference implementation for Distributed Ranges.

`Overview slides`_

.. _`Overview slides`: doc/Distributed%20Ranges.pdf

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
  cmake -B build -DCMAKE_INSTALL_PREFIX=./dr_root
  make -C build install

Use ``-I`` with the path to dr/build/install To find the header files
during compilation::

  g++ -I$PATH_TO_DR/dr_root/include file.cpp


Developer Information
=====================

Print types at compile time: `Print Type`_

Submitting a PR
---------------

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

To fix a problem with a missing license do::

  reuse annotate --exclude-year --license BSD-3-Clause --copyright "Intel Corporation" <filename>

Or copy the license from the top of a file with the same filename
extension.

Editing the Spec
----------------

Build the spec::

  make -C doc/spec html

Open in your browser: ``doc/spec/build/html/index.html``

Doxygen html is at: ``doc/spec/build/doxygen-html/index.html``

There are pre-commit checks for spelling and broken links. To run it manually::

  make -C doc/spec linkcheck
  make -C doc/spec spelling SPHINXOPTS=-q

The `SPHINXOPTS` is necessary to disable warning as errors, so you can
see all the spelling errors instead of the first one. Add spelling
exceptions to `spelling_wordlist.txt`. Do not add variable, class,
function, etc to the exceptions. Spellcheck ignores them if they are
properly delimited in the source doc.

Fuzz Test
---------

Build the test::

  CXX=clang++ cmake -B build
  cd build/fuzz/cpu
  make -j
  ./cpu-fuzz -max_len=16

The command asserts when it finds an error. Otherwise it runs forever
so kill it to stop testing. When it finds an error, it writes the
input to a file in the current directory. To run again for just that
input::

  ./cpu-fuzz . .



.. _`Print Type`: https://stackoverflow.com/a/14617848/2525421
.. _`latest spec`: https://stunning-fortnight-c2e7e025.pages.github.io/spec
.. _`latest doxygen`: https://stunning-fortnight-c2e7e025.pages.github.io/doxygen
