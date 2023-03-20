.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _transform:

===============
 ``transform``
===============

Interface
=========

SHP
---

.. doxygenfunction:: transform(ExecutionPolicy &&policy, lib::distributed_range auto &&in, lib::distributed_iterator auto out, auto &&fn)
   :outline:


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
