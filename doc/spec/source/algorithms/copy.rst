.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _copy:

==========
 ``copy``
==========

Interface
=========

.. doxygenfunction:: lib::copy(DI first, DI last, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(mpi_distributed_contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, contiguous_iterator_or_nullptr auto first, std::size_t size, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, IN first, IN last, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, rng::contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, DI first, DI last, contiguous_iterator_or_nullptr auto result)
.. doxygenfunction:: lib::copy(int root, mpi_distributed_contiguous_iterator auto first, std::size_t size, contiguous_iterator_or_nullptr auto result)
.. doxygenfunction:: lib::copy(int root, mpi_distributed_contiguous_range auto &&r, contiguous_iterator_or_nullptr auto result)

Description
===========

.. seealso::

   `std::ranges::copy`_
     Standard C++ algorithm
   `std::copy`_
     Standard C++ algorithm

Examples
========
