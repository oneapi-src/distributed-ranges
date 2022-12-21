.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _distributed_mdspan:

======================
``distributed_mdspan``
======================

Interface
=========

.. doxygenclass:: lib::distributed_mdspan
   :members:

Description
===========

``distributed_mdspan`` is a generalization of `std::mdspan`_ that
provides non-owning, multi-dimensional indexing of objects where the
objects may be distributed across multiple processes.

.. seealso::

   `std::mdspan`_

   :ref:`distributed_mdarray`

Examples
========
