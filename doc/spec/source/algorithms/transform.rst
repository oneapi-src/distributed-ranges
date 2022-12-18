.. include:: ../include/distributed-ranges.rst

.. _transform:

===============
 ``transform``
===============

Interface
=========

.. doxygenfunction:: lib::transform(DI first, DI last, mpi_distributed_contiguous_iterator auto result, auto op)
.. doxygenfunction:: lib::transform(mpi_distributed_contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result, auto op)
.. doxygenfunction:: lib::transform(DI first1, DI last1, mpi_distributed_contiguous_iterator auto first2, mpi_distributed_contiguous_iterator auto result, auto op)
.. doxygenfunction:: lib::transform(mpi_distributed_contiguous_range auto &&r1, mpi_distributed_contiguous_range auto &&r2, mpi_distributed_contiguous_iterator auto result, auto op)
.. doxygenfunction:: lib::transform(SDI first, SDI last, sycl_mpi_distributed_contiguous_iterator auto result, auto op)


Description
===========

.. seealso::

   `std::transform`_
     C++ model
   `std::ranges::transform`_
     C++ range-based model
   :ref:`reduce`
     related algorithm
   :ref:`transform_reduce`
     related algorithm

Usage
=====
