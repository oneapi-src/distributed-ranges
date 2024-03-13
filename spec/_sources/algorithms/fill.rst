.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _fill:

==========
 ``fill``
==========

Interface
=========

MHP
---

.. doxygenfunction:: dr::mhp::fill(dr::distributed_contiguous_range auto &&dr, auto value);
   :outline:
.. doxygenfunction:: dr::mhp::fill(DI first, DI last, auto value)
   :outline:

SHP
---

.. doxygenfunction:: dr::shp::fill_async(Iter first, Iter last, const std::iter_value_t<Iter> &value)
   :outline:
.. doxygenfunction:: dr::shp::fill(Iter first, Iter last, const std::iter_value_t<Iter> &value)
   :outline:
.. doxygenfunction:: dr::shp::fill_async(device_ptr<T> first, device_ptr<T> last, const U &value)
   :outline:
.. doxygenfunction:: dr::shp::fill(device_ptr<T> first, device_ptr<T> last, const U &value)
   :outline:
.. doxygenfunction:: dr::shp::fill_async(R &&r, const T &value)
   :outline:
.. doxygenfunction:: dr::shp::fill(R &&r, const T &value)
   :outline:
.. doxygenfunction:: dr::shp::fill_async(DR &&r, const T &value)
   :outline:
.. doxygenfunction:: dr::shp::fill(DR &&r, const T &value)
   :outline:
.. doxygenfunction:: dr::shp::fill(Iter first, Iter last, const T &value)
   :outline:

Description
===========

.. seealso::

   C++ model
     `std::fill`_
   C++ model
     `std::ranges::fill`_

Usage
=====
