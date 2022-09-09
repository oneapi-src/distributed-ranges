=======================
 ``distributed_span``
=======================

Interface
=========

.. doxygenclass:: distributed_span
   :members:

Description
===========
`distributed_span` is a generalization of `std::span` that refers to a span of
memory that may be distributed across multiple processes in a parallel program.

`distributed_span` takes a range of `remote_span` objects, along with an
accessor, and represents a span of data distributed across the spans.

Examples
========
