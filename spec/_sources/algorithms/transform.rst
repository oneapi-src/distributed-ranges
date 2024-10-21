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

MP
---

.. doxygenfunction:: dr::mp::transform(rng::forward_range auto &&in, dr::distributed_iterator auto out, auto op)
.. doxygenfunction:: dr::mp::transform(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out, auto op)

SP
---

.. doxygenfunction:: dr::sp::transform(ExecutionPolicy &&policy, dr::distributed_range auto &&in, dr::distributed_iterator auto out, auto &&fn)
.. doxygenfunction:: dr::sp::transform(R &&in, Iter out, Fn &&fn)
.. doxygenfunction:: dr::sp::transform(ExecutionPolicy &&policy, Iter1 in_begin, Iter1 in_end, Iter2 out_end, Fn &&fn)
.. doxygenfunction:: dr::sp::transform(Iter1 in_begin, Iter1 in_end, Iter2 out_end, Fn &&fn)


Description
===========

.. seealso::

   `std::transform`_
     C++ model
   `std::ranges::transform`_
     C++ range-based model
   :ref:`reduce`
     related algorithm

Usage
=====
