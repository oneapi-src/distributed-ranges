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

.. doxygenfunction:: dr::shp::flat_gemv(C &&c, dr::shp::sparse_matrix<T, I> &a, B &&b)
.. doxygenfunction:: dr::shp::gemv(C &&c, dr::shp::sparse_matrix<T, I> &a, B &&b, shp::duplicated_vector<rng::range_value_t<B>> &scratch)
.. doxygenfunction:: dr::shp::gemv(C &&c, dr::shp::sparse_matrix<T, I> &a, B &&b)
.. doxygenfunction:: dr::shp::gemv_square(C &&c, dr::shp::sparse_matrix<T, I> &a, B &&b)
.. doxygenfunction:: dr::shp::gemv_square_copy(C &&c, dr::shp::sparse_matrix<T, I> &a, B &&b)

Description
===========


Examples
========
