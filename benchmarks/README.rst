.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

============
 Benchmarks
============

Streams
=======

Used as a baseline for memory bound applications. Assume we are using
48 core machine and want each core to process :math:`10,000,000`
elements of type `double`. To build::

  git clone https://github.com/jeffhammond/STREAM.git
  cd STREAM
  gcc -fopenmp -O3 -mcmodel=medium -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=480000000 -DNTIMES=100 stream.c -o stream_cpu_openmp

When running::

  OMP_NUM_THREADS=48 ./stream_cpu_openmp

To verify that you are using 48 cores, run ``top`` in another
window. The stream process should show 4800 in the ``%CPU`` column.
