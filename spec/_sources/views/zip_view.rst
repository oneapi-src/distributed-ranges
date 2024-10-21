.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause


.. include:: ../include/distributed-ranges.rst

.. _zip_view:

=============
 ``zip_view``
=============

Interface
=========

MP
---

.. doxygenclass:: dr::mp::zip_view
   :members:
.. doxygenfunction:: dr::mp::views::zip
   :outline:

SP
---

.. doxygenclass:: dr::sp::zip_view
   :members:
.. doxygenfunction:: dr::sp::views::zip
   :outline:

Description
===========

.. seealso::

   `std::ranges::views::zip`_
     Standard C++ view
