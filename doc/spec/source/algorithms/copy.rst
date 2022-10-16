==========
 ``copy``
==========

Interface
=========

.. doxygenfunction:: copy(ExecutionPolicy e, R &&r, O result)
.. doxygenfunction:: copy(device_ptr<const T> first, device_ptr<const T> last, T *d_first)
.. doxygenfunction:: copy(const T *first, const T *last, device_ptr<T> d_first)

Description
===========

.. seealso:: `std::ranges::copy <https://en.cppreference.com/w/cpp/algorithm/ranges/copy>`__

Examples
========
