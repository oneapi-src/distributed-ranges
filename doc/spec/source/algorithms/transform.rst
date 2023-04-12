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

.. doxygenfunction:: transform(ExecutionPolicy &&policy, dr::distributed_range auto &&in, dr::distributed_iterator auto out, auto &&fn)


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
