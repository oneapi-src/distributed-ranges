=======================
 ``distributed_span``
=======================

Interface
=========

.. doxygenclass:: lib::distributed_span
   :members:

Description
===========

``distributed_span`` is a generalization of ``std::span`` that provides
a view of a contiguous span of memory distributed across multiple processes
in a parallel program.

``distributed_span`` takes a range of ``remote_span`` objects,
along with an accessor, and represents a span of data distributed
across the spans.

.. seealso::

   :ref:`remote_span`

   `std::span <https://en.cppreference.com/w/cpp/container/span>`__


Examples
========
