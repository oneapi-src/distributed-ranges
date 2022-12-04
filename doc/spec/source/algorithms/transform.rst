.. include:: ../include/distributed-ranges.rst

.. _transform:

===============
 ``transform``
===============

Interface
=========

.. doxygenfunction:: lib::transform(typename DistObj::const_iterator input_iterator, typename DistObj::const_iterator sentinel, typename DistObj::iterator output_iterator, UnaryOp op)
.. doxygenfunction:: lib::transform(R &&input_range, OutputIterator output_iterator, UnaryOp op)
.. doxygenfunction:: lib::transform(typename DistObj::iterator first1, typename DistObj::iterator last1, typename DistObj::iterator first2, typename DistObj::iterator d_first, BinaryOp op)
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
