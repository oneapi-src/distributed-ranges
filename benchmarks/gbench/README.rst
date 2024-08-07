.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

==============
 Google Bench
==============

We use google bench for micro-benchmarks.

Show standard google bench options::

  ./mp-bench --help

Show custom options::

  ./mp-bench --drhelp

See `user guide`_ for more information on google benchmark.

MP Sample Commands
===================

Run all benchmarks with 2 ranks. Each rank uses a single thread::

  mpirun -n 2 ./mp-bench --benchmark_counters_tabular=true

Run all benchmarks with 2 ranks. Each rank uses a single SYCL device::

  mpirun -n 2 ./mp-bench --benchmark_counters_tabular=true --sycl

Run 2D stencil algorithms::

  mpirun -n 2 ./mp-bench --benchmark_counters_tabular=true --benchmark_filter=Stencil2D

Run distributed ranges algorithms::

  mpirun -n 2 ./mp-bench --benchmark_counters_tabular=true --benchmark_filter=.*DR


SP Sample Commands
===================

By default, SP uses all available devices. When running on a 2 socket
CPU system, SP partitions the root device into 2 devices. Use ``-d``
to explicitly control the number of devices::

  ./sp-bench --benchmark_time_unit=ms --benchmark_counters_tabular=true -d 2

Benchmark Variants
==================

DR
  distributed ranges
Serial
  single thread
SYCL
  single SYCL device, direct coded in SYCL
DPL
  single SYCL device using DPL


.. _`user guide`: https://github.com/google/benchmark/blob/main/docs/user_guide.md#custom-counters
