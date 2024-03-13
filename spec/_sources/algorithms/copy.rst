.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _copy:

==========
 ``copy``
==========

Interface
=========

MHP
---

.. doxygenfunction:: dr::mhp::copy(rng::forward_range auto &&in, dr::distributed_iterator auto out)
   :outline:
.. doxygenfunction:: dr::mhp::copy(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out)
   :outline:
.. doxygenfunction:: dr::mhp::copy(std::size_t root, dr::distributed_contiguous_range auto &&in, std::contiguous_iterator auto out)
   :outline:
.. doxygenfunction:: dr::mhp::copy(std::size_t root, rng::contiguous_range auto &&in, dr::distributed_contiguous_iterator auto out)
   :outline:

SHP
---

.. doxygenfunction:: dr::shp::copy(InputIt first, InputIt last, OutputIt d_first)
   :outline:
.. doxygenfunction:: dr::shp::copy(device_ptr<T> first, device_ptr<T> last, Iter d_first)
   :outline:


Description
===========

.. seealso::

   `std::ranges::copy`_
     Standard C++ algorithm
   `std::copy`_
     Standard C++ algorithm
