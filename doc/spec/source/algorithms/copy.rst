.. include:: ../include/distributed-ranges.rst

.. _copy:

==========
 ``copy``
==========

Interface
=========

.. doxygenfunction:: lib::collective::copy(int root, R &&src, distributed_vector<rng::range_value_t<R>> &dst)
.. doxygenfunction:: lib::collective::copy(int root, distributed_vector<rng::range_value_t<R>> &src, R &&dst)
.. doxygenfunction:: lib::copy(mpi_distributed_contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(I first, I last, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, rng::contiguous_range auto &&r, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, I first, I last, mpi_distributed_contiguous_iterator auto result)
.. doxygenfunction:: lib::copy(int root, mpi_distributed_contiguous_range auto &&r, auto result)
.. doxygenfunction:: lib::copy(int root, I first, I last, auto result)

Description
===========

.. seealso:: `std::ranges::copy`_

Examples
========
