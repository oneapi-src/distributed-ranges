.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

==============
 Google Bench
==============

We use google bench for micro-benchmarks.

Sample Commands
===============

Run with all benchmarks with 2 ranks::

  mpirun -n 2 ./mhp-bench --benchmark_time_unit=ms --benchmark_counters_tabular=true

Run 2D stencil algorithms::

  mpirun -n 2 ./mhp-bench --benchmark_time_unit=ms --benchmark_counters_tabular=true --benchmark_filter=Stencil2D

Run distributed ranges algorithms::

  mpirun -n 2 ./mhp-bench --benchmark_time_unit=ms --benchmark_counters_tabular=true --benchmark_filter=.*DR

Show standard google bench options::

  ./mhp-bench --help

Show custom options::

  ./mhp-bench --drhelp

See `user guide`_ for more information on google benchmark.

Details
=======

On a 2 socket system without GPU's, SHP partitions the single SYCL CPU
device by affinity domain into 2 subdevices and uses both. MHP only
uses the root device. For fair comparison, you must use 2 MHP
processes.


.. _`user guide`: https://github.com/google/benchmark/blob/main/docs/user_guide.md#custom-counters
