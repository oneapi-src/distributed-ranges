.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _for_each:

==============
 ``for_each``
==============

Interface
=========

.. doxygenfunction:: for_each(DI first, DI last, auto op)
.. doxygenfunction:: for_each(mpi_distributed_contiguous_range auto &&r, auto op)

Description
===========

.. seealso::

   `std::ranges::for_each`_
     Standard C++ algorithm
   `std::for_each`_
     Standard C++ algorithm

Examples
========
