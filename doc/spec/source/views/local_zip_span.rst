.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _local_zip_span:

====================
 ``local_zip_span``
====================

Interface
=========

.. doxygenclass:: lib::view_local_zip_span
   :members:
.. doxygenstruct:: lib::local_zip_span
   :members:

Description
===========

Range adaptor to reference local elements of a range that is a zip of
multiple distributed ranges.

Examples
========
