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

.. doxygenfunction:: mhp::copy(dr::distributed_contiguous_range auto &&in, dr::distributed_iterator auto out)
   :outline:
.. doxygenfunction:: mhp::copy(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out)
   :outline:

SHP
---

.. doxygenfunction:: shp::copy(InputIt first, InputIt last, OutputIt d_first)
   :outline:
.. doxygenfunction:: shp::copy(device_ptr<T> first, device_ptr<T> last, Iter d_first)
   :outline:


Description
===========

.. seealso::

   `std::ranges::copy`_
     Standard C++ algorithm
   `std::copy`_
     Standard C++ algorithm
