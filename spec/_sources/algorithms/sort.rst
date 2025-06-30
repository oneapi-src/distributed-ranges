.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _sort:

==========
 ``sort``
==========

Interface
=========

MP
---

SP
---

.. doxygenfunction:: dr::sp::sort(R &&r, Compare comp = Compare())
   :outline:
.. doxygenfunction:: dr::sp::sort(RandomIt first, RandomIt last, Compare comp = Compare())
   :outline:

Description
===========

.. seealso::

   C++ model
     `std::sort`_
   C++ model
     `std::ranges::sort`_

Usage
=====
