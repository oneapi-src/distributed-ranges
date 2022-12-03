.. include:: ../include/distributed-ranges.rst

.. _reduce:

============
 ``reduce``
============

Interface
=========

.. doxygenfunction:: lib::reduce(int root, R &&r, T init, BinaryOp &&binary_op)
.. doxygenfunction:: lib::reduce(int root, I input_iterator, S sentinel, T init, BinaryOp &&binary_op)

Description
===========

.. seealso:: `std::reduce`_

Examples
========
