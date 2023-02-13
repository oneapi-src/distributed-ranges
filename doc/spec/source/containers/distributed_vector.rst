.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _distributed_vector:

======================
``distributed_vector``
======================

Interface
=========

.. doxygenclass:: lib::distributed_vector
   :members:

Description
===========

``distributed_vector`` is a generalization of `std::vector`_ that
provides a sequence of objects where the objects may be distributed
across multiple processes.

.. seealso::

   `std::vector`_

   :ref:`halo_bounds`


Usage
=====
