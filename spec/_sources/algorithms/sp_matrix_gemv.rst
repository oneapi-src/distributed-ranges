.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. include:: ../include/distributed-ranges.rst

.. _gemv:

======================
 ``gemv``
======================

Interface
=========

.. doxygenfunction:: dr::sp::flat_gemv(C &&c, dr::sp::sparse_matrix<T, I> &a, B &&b)
.. doxygenfunction:: dr::sp::gemv(C &&c, dr::sp::sparse_matrix<T, I> &a, B &&b, sp::duplicated_vector<rng::range_value_t<B>> &scratch)
.. doxygenfunction:: dr::sp::gemv(C &&c, dr::sp::sparse_matrix<T, I> &a, B &&b)
.. doxygenfunction:: dr::sp::gemv_square(C &&c, dr::sp::sparse_matrix<T, I> &a, B &&b)
.. doxygenfunction:: dr::sp::gemv_square_copy(C &&c, dr::sp::sparse_matrix<T, I> &a, B &&b)

Description
===========


Examples
========
