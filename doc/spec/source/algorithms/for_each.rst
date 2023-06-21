.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _for_each:

==============
 ``for_each``
==============

Interface
=========
MHP
---
.. doxygenfunction:: dr::mhp::for_each(dr::distributed_range auto &&dr, auto op)
  :outline:
.. doxygenfunction:: dr::mhp::for_each(DI first, DI last, auto op)
  :outline:

SHP
---

.. doxygenfunction:: dr::shp::for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn)
  :outline:
.. doxygenfunction:: dr::shp::for_each(ExecutionPolicy &&policy, Iter begin, Iter end, Fn &&fn)
  :outline:
.. doxygenfunction:: dr::shp::for_each(R &&r, Fn &&fn)
  :outline:
.. doxygenfunction:: dr::shp::for_each(Iter begin, Iter end, Fn &&fn)
  :outline:

Description
===========

.. seealso::

   `std::ranges::for_each`_
     Standard C++ algorithm
   `std::for_each`_
     Standard C++ algorithm

Examples
========
