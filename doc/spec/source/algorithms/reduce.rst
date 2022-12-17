.. include:: ../include/distributed-ranges.rst

.. _reduce:

============
 ``reduce``
============

Interface
=========

.. doxygenfunction:: lib::reduce(int root, mpi_distributed_contiguous_range auto &&r, T init, auto &&binary_op)
.. doxygenfunction:: lib::reduce(int root, I first, I last, T init, auto &&binary_op)
.. doxygenfunction:: lib::reduce(int root, I first, I last, T init, BinaryOp &&binary_op)

Description
===========

.. seealso:: `std::reduce`_

Examples
========
