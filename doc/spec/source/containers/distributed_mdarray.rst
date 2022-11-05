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

   * :ref:`distributed_mdspan`
   * `std::mdarray`_


.. _`std::mdarray`: https://www.open-std.org/JTC1/SC22/WG21/docs/papers/2022/p1684r2.html

Examples
========
