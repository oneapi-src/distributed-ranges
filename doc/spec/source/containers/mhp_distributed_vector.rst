.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _mhp_distributed_vector:

===============================
``dr::mhp::distributed_vector``
===============================

Interface
=========

.. doxygenclass:: dr::mhp::distributed_vector
   :members:

Description
===========

Vector distributed among MPI nodes, with support
for data exchange at segment edges (halo)

.. seealso::

   `std::vector`_
