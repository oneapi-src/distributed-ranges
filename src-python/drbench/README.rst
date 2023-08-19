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
inspect.  Devcloud runner dry run::

  dr-bench suite --no-p2p --gpus 4 --sockets 2 --cores-per-socket 56 --dry-run

Aurora runner dry run, run 1-12 gpus::

  dr-bench suite --gpus 12 --dry-run

Run 1-4 nodes, using 12 gpus::

  dr-bench suite --nodes 4 --gpus 12 --dry-run

To test the plotter, use a set of json files that is published as an
artifact in CI. Select a run from the `Devcloud runs`_. Artifacts are
listed at the bottom, download and unzip the file.

.. _`Devcloud runs`: https://github.com/oneapi-src/distributed-ranges/actions/workflows/devcloud.yml
