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

MP
---

.. doxygenfunction:: dr::mp::iota(R &&r, T value)
  :outline:
.. doxygenfunction:: dr::mp::iota(Iter begin, Iter end, T value)
  :outline:

SP
---

.. doxygenfunction:: dr::sp::iota(R &&r, T value)
  :outline:
.. doxygenfunction:: dr::sp::iota(Iter begin, Iter end, T value)
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
