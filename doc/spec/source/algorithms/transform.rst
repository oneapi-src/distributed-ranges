.. include:: ../include/distributed-ranges.rst

.. _transform:

===============
 ``transform``
===============

Interface
=========

.. doxygenfunction:: lib::transform(index_iterator<DistObj> input_iterator, index_iterator<DistObj> sentinel, index_iterator<DistObj> output_iterator, UnaryOp op)
.. doxygenfunction:: lib::transform(R &&input_range, OutputIterator output_iterator, UnaryOp op)
.. doxygenfunction:: lib::transform(index_iterator<DistObj> first1, index_iterator<DistObj> last1, index_iterator<DistObj> first2, index_iterator<DistObj> d_first, BinaryOp op)
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
