.. include:: ../include/distributed-ranges.rst

.. _transform:

===============
 ``transform``
===============

Interface
=========

.. doxygenfunction:: lib::transform(I first, I last, O result, UnaryOp op)
.. doxygenfunction:: lib::transform(R &&r, O result, UnaryOp op)
.. doxygenfunction:: lib::transform(I1 first1, I1 last1, I2 first2, O result, BinaryOp op)
.. doxygenfunction:: lib::transform(R1 &&r1, R2 &&r2, O result, BinaryOp op)


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
