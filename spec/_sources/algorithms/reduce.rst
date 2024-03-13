.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _reduce:

============
 ``reduce``
============

Interface
=========

MHP
---

.. doxygenfunction:: dr::mhp::reduce(std::size_t root, DR &&dr, T init, auto &&binary_op)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(DR &&dr, T init, auto &&binary_op)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(std::size_t root, DR &&dr, T init)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(DR &&dr, T init)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(std::size_t root, DR &&dr)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(DR &&dr)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(std::size_t root, DI first, DI last, T init, auto &&binary_op)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(DI first, DI last, T init, auto &&binary_op)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(std::size_t root, DI first, DI last, T init)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(DI first, DI last, T init)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(std::size_t root, DI first, DI last)
  :outline:
.. doxygenfunction:: dr::mhp::reduce(DI first, DI last)
  :outline:

SHP
---

.. doxygenfunction:: dr::shp::reduce(ExecutionPolicy &&policy, R &&r, T init, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::shp::reduce(ExecutionPolicy &&policy, R &&r, T init)
  :outline:
.. doxygenfunction:: dr::shp::reduce(ExecutionPolicy &&policy, R &&r)
  :outline:

Iterator versions

.. doxygenfunction:: dr::shp::reduce(ExecutionPolicy &&policy, Iter first, Iter last)
  :outline:
.. doxygenfunction:: dr::shp::reduce(ExecutionPolicy &&policy, Iter first, Iter last, T init)
  :outline:
.. doxygenfunction:: dr::shp::reduce(ExecutionPolicy &&policy, Iter first, Iter last, T init, BinaryOp &&binary_op)
  :outline:

Execution policy-less algorithms

.. doxygenfunction:: dr::shp::reduce(R &&r)
  :outline:
.. doxygenfunction:: dr::shp::reduce(R &&r, T init)
  :outline:
.. doxygenfunction:: dr::shp::reduce(R &&r, T init, BinaryOp &&binary_op)
  :outline:

Description
===========

.. seealso:: `std::reduce`_

Examples
========
