.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

==========
 dr-bench
==========

Python package for running benchmarks and plotting results.

Testing
=======

Use ``--dry-run`` to show all the commands it will run and visually
inspect. Run on 4 GPUs and on CPU with 2 sockets and 56 cores per
socket. Run 1-4 GPUs and 1-2 sockets::

  dr-bench suite --gpus 4 --sockets 2 --cores-per-socket 56 --dry-run

Run 1-4 nodes, using 12 gpus::

  dr-bench suite --nodes 4 --gpus 12 --dry-run

To test the plotter, use a set of json files that is published as an
artifact in CI. You can also run manually to generate the data::

  # speedup plots need a reference
  dr-bench run --target mhp_sycl_gpu --ranks 1 -f BlackScholes_Reference --mhp-bench ./mhp-quick-bench --device-memory
  # run 1-4 GPUs
  dr-bench run --target mhp_sycl_gpu --rank-range 4 -f BlackScholes_DR --mhp-bench ./mhp-quick-bench --device-memory

Make all the plots that have data::

  dr-bench plot
