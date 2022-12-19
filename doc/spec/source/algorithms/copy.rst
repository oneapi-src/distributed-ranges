.. include:: ../include/distributed-ranges.rst

.. _copy:

==========
 ``copy``
==========

Interface
=========

.. doxygenfunction:: lib::copy(DI first, DI last, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(mpi_distributed_contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, I first, I last, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, I first, std::size_t size, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, rng::contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, DI first, DI last, std::contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, DI first, std::size_t size, std::contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, mpi_distributed_contiguous_range auto &&r, auto result)

Description
===========

.. seealso::

   `std::ranges::copy`_
     Standard C++ algorithm
   `std::copy`_
     Standard C++ algorithm

Examples
========
