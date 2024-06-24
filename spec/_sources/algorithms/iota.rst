.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _iota:

==========
 ``iota``
==========

Interface
=========

MHP
---

.. doxygenfunction:: dr::mhp::iota(R &&r, T value)
  :outline:
.. doxygenfunction:: dr::mhp::iota(Iter begin, Iter end, T value)
  :outline:

SHP
---

.. doxygenfunction:: dr::shp::iota(R &&r, T value)
  :outline:
.. doxygenfunction:: dr::shp::iota(Iter begin, Iter end, T value)
  :outline:



Description
===========

.. seealso::

   `std::ranges::iota`_
     Standard C++ algorithm
   `std::iota`_
     Standard C++ algorithm

Usage
=====
