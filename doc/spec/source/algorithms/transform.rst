.. include:: ../include/distributed-ranges.rst

.. _transform:

===============
 ``transform``
===============

Interface
=========

.. doxygenfunction:: lib::transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp op)
.. doxygenfunction:: lib::transform(R &&input_range, OutputIterator output_iterator, UnaryOp op)
.. doxygenfunction:: lib::transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp op)
.. doxygenfunction:: lib::transform(R1 &&r1, R2 &&r2, O output, BinaryOp op)


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
