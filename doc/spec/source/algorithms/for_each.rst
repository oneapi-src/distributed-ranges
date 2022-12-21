.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _for_each:

==============
 ``for_each``
==============

Interface
=========

.. doxygenfunction:: for_each(ExecutionPolicy &&policy, R range, UnaryFunction f)
.. doxygenfunction:: for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn)

Description
===========

.. seealso:: `std::for_each`_

Examples
========
