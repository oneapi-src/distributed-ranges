.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _inclusive_scan:

============================
 ``inclusive_scan``
============================

Interface
=========

MP
---

SP
---

.. doxygenfunction:: dr::sp::inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, BinaryOp &&binary_op, T init)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last, OutputIter d_first, BinaryOp &&binary_op, T init)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last, OutputIter d_first, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last, OutputIter d_first)
  :outline:

Execution policy-less versions

.. doxygenfunction:: dr::sp::inclusive_scan(R &&r, O &&o)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op, T init)
  :outline:

Distributed iterator versions

.. doxygenfunction:: dr::sp::inclusive_scan(Iter first, Iter last, OutputIter d_first)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(Iter first, Iter last, OutputIter d_first, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::inclusive_scan(Iter first, Iter last, OutputIter d_first, BinaryOp &&binary_op, T init)
  :outline:

Description
===========

.. seealso::

  'std::inclusive_scan'_

Examples
========
