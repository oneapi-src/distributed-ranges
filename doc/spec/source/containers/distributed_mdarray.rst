.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _distributed_mdarray:

=======================
``distributed_mdarray``
=======================

Interface
=========

.. doxygenclass:: lib::distributed_mdarray
   :members:

Description
===========

``distributed_mdarray`` is a generalization of `std::mdarray`_ that
provides a multi-dimensional indexing of objects where the objects may
be distributed across multiple processes.

.. seealso::

   `std::mdarray`_
     C++ model
   :ref:`distributed_mdspan`
     View for a ``distributed_mdarray``

Usage
=====
