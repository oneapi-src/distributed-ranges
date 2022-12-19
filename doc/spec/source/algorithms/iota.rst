.. include:: ../include/distributed-ranges.rst

.. _iota:

==========
 ``iota``
==========

Interface
=========

.. doxygenfunction:: lib::iota(DI first, DI last, auto value)
.. doxygenfunction:: lib::iota(mpi_distributed_contiguous_range auto &&r, auto value)

Description
===========

.. seealso::

   `std::ranges::iota`_
     Standard C++ algorithm
   `std::iota`_
     Standard C++ algorithm

Usage
=====
