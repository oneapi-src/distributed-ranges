.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _remote_vector:

=================
``remote_vector``
=================

Interface
=========

.. doxygenclass:: lib::remote_vector
   :members:

Description
===========

``remote_vector`` is a generalization of ``std::vector`` that provides
a sequence of objects where the objects reside in a single process and
can be accessed by an process in the owning team.


.. seealso::

   `std::vector`_

Examples
========
