## DISCONTINUATION OF PROJECT. 
This project will no longer be maintained by Intel. 

Intel will not provide or guarantee development of or support for this project, including but not limited to, maintenance, bug fixes, new releases or updates. Patches to this project are no longer accepted by Intel.

This project has been identified as having known security issues. 

Contact: webadmin@linux.intel.com
.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===================
 Distributed Ranges
===================

.. image:: https://github.com/oneapi-src/distributed-ranges/actions/workflows/pr.yml/badge.svg
   :target: https://github.com/oneapi-src/distributed-ranges/actions/workflows/pr.yml
.. image:: https://www.bestpractices.dev/projects/8975/badge
   :target: https://www.bestpractices.dev/projects/8975

Productivity library for distributed and partitioned memory based on
C++ Ranges.

About
-----

Distributed Ranges is a C++ productivity library for distributed and partitioned memory based on C++ ranges.
It offers a collection of data structures, views, and algorithms for building generic abstractions
and provides interoperability with MPI, SHMEM, SYCL and OpenMP and portability on CPUs and GPUs.
NUMA-aware allocators and distributed data structures facilitate development of C++ applications
on heterogeneous nodes with multiple devices and achieve excellent performance and parallel scalability
by exploiting local compute and data access.

Main strength of the library
============================

In this model one can:

* create a `distributed data structure` that work with all our algorithms out of the box
* create an `algorithm` that works with all our distributed data structures out of the box

Distributed Ranges is a `glue` that makes this possible.


Documentation
-------------

* Usage:

  * Introductory presentation: `Distributed Ranges, why you need it`_, 2024
  * Article: `Get Started with Distributed Ranges`_, 2023
  * Tutorial: `Distributed Ranges Tutorial`_

* Design / Implementation:

  * Conference paper: `Distributed Ranges, A Model for Distributed Data Structures, Algorithms, and Views`_, 2024
  * Talk: `CppCon 2023; Benjamin Brock; Distributed Ranges`_, 2023
  * Technical presentation: `Intel Innovation'23`_, 2023
  * `API specification`_


Requirements
------------

* Linux
* cmake >=3.20
* `OneAPI HPC Toolkit`_ installed

Enable `OneAPI` by::

  source ~/intel/oneapi/setvars.sh

... or by::

  source /opt/intel/oneapi/setvars.sh

... or wherever you have ``oneapi/setvars.sh`` script installed in your system.

Additional requirements for NVIDIA GPUs
=======================================

* `CUDA`_
* `OneAPI for NVIDIA GPUs`_ plugin

When enabling OneAPI use ``--include-intel-llvm`` option, e.g. call::

  source ~/intel/oneapi/setvars.sh --include-intel-llvm

... instead of ``source ~/intel/oneapi/setvars.sh``.


Build and run
-------------

Build for Intel GPU/CPU
=======================

All tests and examples can be build by::

  CXX=icpx cmake -B build
  cmake --build build -- -j


Build for NVidia GPU
====================

.. note::

  Distributed Ranges library works in two models:
   - Multi Process (based on SYCL and MPI)
   - Single Process (based on pure SYCL)

  On NVIDIA GPU only `Multi Process` model is currently supported.

To build multi-process tests call::

  CXX=icpx cmake -B build -DENABLE_CUDA:BOOL=ON
  cmake --build build --target mp-all-tests -- -j


Run tests
=========

Run multi process tests::

  ctest --test-dir build --output-on-failure -L MP -j 4

Run single process tests::

  ctest --test-dir build --output-on-failure -L SP -j 4

Run all tests::

  ctest --test-dir build --output-on-failure -L TESTLABEL -j 4

Run benchmarks
==============

Two binaries are build for benchmarks:

* mp-bench - for benchmarking `Multi-Process` model
* sp-bench - for benchmarking `Single-Process` model

Here are examples of running single benchmarks.

Running `GemvEq_DR` strong scaling benchmark in Multi-Process model using two GPUs::

  ONEAPI_DEVICE_SELECTOR='level_zero:gpu' I_MPI_OFFLOAD=1 I_MPI_OFFLOAD_CELL_LIST=0-11 \
  mpiexec -n 2 -ppn 2  build/benchmarks/gbench/mp/mp-bench --vector-size 1000000000 --reps 50 \
  --v=3 --benchmark_out=mp_gemv.txt --benchmark_filter=GemvEq_DR/ --sycl

Running `Exclusive_Scan_DR` weak scaling in Single-Process model using two GPUs::

  ONEAPI_DEVICE_SELECTOR='level_zero:gpu' KMP_AFFINITY=compact \
  build/benchmarks/gbench/sp/sp-bench --vector-size 1000000000 --reps 50 \
  --v=3 --benchmark_out=sp_exclscan.txt --benchmark_filter=Exclusive_Scan_DR/ \
  --weak-scaling --device-memory --num-devices 2


Check all options::

  ./build/benchmarks/gbench/mp/mp-bench --help  # see google test options help
  ./build/benchmarks/gbench/mp/mp-bench --drhelp  # see DR specific options



Examples
--------

See `Distributed Ranges Tutorial`_ for a few well explained examples.

Adding Distributed Ranges to your project
-----------------------------------------

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

See `Distributed Ranges Tutorial`_
for a live example of a cmake project that imports and uses Distributed Ranges.

Logging
-------

Add below code to your ``main`` function to enable logging.

If using `Single-Process` model::

  std::ofstream logfile("dr.log");
  dr::drlog.set_file(logfile);

If using `Multi-Process` model::

  int my_mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);
  std::ofstream logfile(fmt::format("dr.{}.log", my_mpi_rank));

Example of adding custom log statement to your code::

  DRLOG("my debug message with varA:{} and varB:{}", a, b);


Contact us
----------

Contact us by writing a `new issue`_.

We seek collaboration opportunities and welcome feedback on ways to extend the library,
according to developer needs.


See also
--------

* `CONTRIBUTING`_
* `Fuzz Testing`_
* `Spec Editing`_ - Editing the API document
* `Print Type`_ - Print types at compile time:
* `Testing`_ - Test system maintenance
* `Security`_ - Security policy
* `Doxygen`_

.. _`Security`: SECURITY.md
.. _`Testing`: doc/developer/testing
.. _`Spec Editing`: doc/spec/README.rst
.. _`Fuzz Testing`: test/fuzz/README.rst
.. _`Print Type`: https://stackoverflow.com/a/14617848/2525421
.. _`CONTRIBUTING`: CONTRIBUTING.md
.. _`Distributed Ranges, why you need it`: https://github.com/oneapi-src/distributed-ranges/blob/main/doc/presentations/Distributed%20Ranges%2C%20why%20you%20need%20it.pdf
.. _`Get Started with Distributed Ranges`: https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-distributed-ranges.html
.. _`Distributed Ranges Tutorial`: https://github.com/oneapi-src/distributed-ranges-tutorial
.. _`Distributed Ranges, A Model for Distributed Data Structures, Algorithms, and Views`: https://dl.acm.org/doi/10.1145/3650200.3656632
.. _`CppCon 2023; Benjamin Brock; Distributed Ranges`: https://www.youtube.com/watch?v=X_dlJcV21YI
.. _`Intel Innovation'23`: https://github.com/oneapi-src/distributed-ranges/blob/main/doc/presentations/Distributed%20Ranges.pdf
.. _`API specification`: https://oneapi-src.github.io/distributed-ranges/spec/
.. _`Doxygen`: https://oneapi-src.github.io/distributed-ranges/doxygen/
.. _`new issue`: issues/new
.. _`OneAPI HPC Toolkit`: https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html
.. _`OneAPI for NVIDIA GPUs`: https://developer.codeplay.com/products/oneapi/nvidia/home/
.. _`CUDA`: https://developer.nvidia.com/cuda-toolkit
