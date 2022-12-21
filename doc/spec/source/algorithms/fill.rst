.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _fill:

==========
 ``fill``
==========

Interface
=========

.. doxygenfunction:: lib::fill(DI first, DI last, auto value)
.. doxygenfunction:: lib::fill(mpi_distributed_contiguous_range auto &&r, auto value)

Description
===========

.. seealso::

   C++ model
     `std::fill`_
   C++ model
     `std::ranges::fill`_

Usage
=====
