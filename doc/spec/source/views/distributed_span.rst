.. include:: ../include/distributed-ranges.rst

.. _distributed_span:

=======================
 ``distributed_span``
=======================

Interface
=========

.. doxygenclass:: lib::distributed_span
   :members:

Description
===========

``distributed_span`` is a generalization of `std::span`_ that provides
a view of a contiguous span of memory distributed across multiple processes
in a parallel program.

``distributed_span`` takes a range of :ref:`remote_span` objects,
along with an accessor, and represents a span of data distributed
across the spans.

.. seealso::

   :ref:`remote_span`

   `std::span`_


Examples
========
