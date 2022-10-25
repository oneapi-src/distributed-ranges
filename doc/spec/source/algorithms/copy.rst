==========
 ``copy``
==========

Interface
=========

.. doxygenfunction:: lib::collective::copy(int root, R &&src, distributed_vector<rng::range_value_t<R>> &dst)
.. doxygenfunction:: lib::collective::copy(int root, distributed_vector<rng::range_value_t<R>> &src, R &&dst)

Description
===========

.. seealso:: `std::ranges::copy <https://en.cppreference.com/w/cpp/algorithm/ranges/copy>`__

Examples
========
