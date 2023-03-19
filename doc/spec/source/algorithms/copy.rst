.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _copy:

==========
 ``copy``
==========

Synopsis
========

MHP
---

.. doxygenfunction::copy(lib::distributed_contiguous_range auto &&in, lib::distributed_iterator auto out)
.. doxygenfunction::copy(DI_IN &&first, DI_IN &&last, lib::distributed_iterator auto &&out)

SHP
---


Description
===========

.. seealso::

   `std::ranges::copy`_
     Standard C++ algorithm
   `std::copy`_
     Standard C++ algorithm

Examples
========
