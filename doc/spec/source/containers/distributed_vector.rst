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

   :ref:`stencil`


Usage
=====
