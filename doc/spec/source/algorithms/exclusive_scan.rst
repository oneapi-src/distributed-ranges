.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _exclusive_scan:

============================
 ``exclusive_scan``
============================

Interface
=========
MP
---

SP
---

.. doxygenfunction:: dr::sp::exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(R &&r, O &&o, T init, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(R &&r, O &&o, T init)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last, OutputIter d_first, T init, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last, OutputIter d_first, T init)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(Iter first, Iter last, OutputIter d_first, T init, BinaryOp &&binary_op)
  :outline:
.. doxygenfunction:: dr::sp::exclusive_scan(Iter first, Iter last, OutputIter d_first, T init)
  :outline

Description
===========

.. seealso::

  'std::exclusive_scan'_

Examples
========
