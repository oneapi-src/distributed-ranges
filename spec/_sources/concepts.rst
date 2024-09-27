.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _concepts:

========
Concepts
========

``remote_iterator``
===================

.. doxygenconcept:: dr::remote_iterator

Defined in ``concepts.hpp``
::

  template <typename I>
  concept remote_iterator =
    std::forward_iterator<I> && requires(I &iter) { dr::ranges::rank(iter); };

Requirements
""""""""""""
1. ``I`` fulfills ``std::forward_iterator``
2. ``i`` has a method ``rank`` returning the rank on which the memory
   ``i`` references is located.

``remote_range``
================

.. doxygenconcept:: dr::remote_range

Defined in ``concepts.hpp``
::

  template <typename R>
  concept remote_range =
    rng::forward_range<R> && requires(R &r) { dr::ranges::rank(r); };

Requirements
""""""""""""
1. ``R`` fulfills ``rng::forward_range``
2. ``r`` has a method ``rank`` returning the rank on which the memory
   ``r`` references is located.

``distributed_range``
=====================

.. doxygenconcept:: dr::distributed_range

Defined in ``concepts.hpp``
::

  template <typename R>
  concept distributed_range =
    rng::forward_range<R> && requires(R &r) { dr::ranges::segments(r); };

Requirements
""""""""""""
1. ``R`` fulfills ``rng::forward_range``
2. ``r`` has a method ``segments``

``remote_contiguous_iterator``
==============================

.. doxygenconcept:: dr::remote_contiguous_iterator

A remote contiguous iterator acts as a pointer to some contiguous piece
of remote memory.

Defined in ``concepts.hpp``
::

  template <typename I>
  concept remote_contiguous_iterator = std::random_access_iterator<I> &&
      requires(I i) {
    { i.rank() } -> std::convertible_to<std::size_t>;
    { i.local() } -> std::contiguous_iterator;
  };


Requirements
""""""""""""

An object `i` of type ``I`` fulfills ``remote_contiguous_iterator``
if and only if:

1. ``I`` fulfills ``std::random_access_iterator``
2. ``i`` has a method ``rank`` returning the rank on which the memory
   ``i`` references is located.
3. ``i`` has a method ``local`` returning an object ``l`` whose type
   fulfills ``std::contiguous_iterator``.  Dereferencing ``l`` is
   well-defined if the current rank is equal ``i.rank()``.

Remarks
"""""""
Instantiations of `remote_ptr`, `device_ptr`, and `BCL::GlobalPtr` should all
fulfill ``remote_contiguous_iterator``.


``remote_contiguous_range``
===========================

.. doxygenconcept:: dr::remote_contiguous_range

A remote contiguous range is a range located in a contiguous piece of remote
memory.

Defined in ``concepts.hpp``

::

  template <typename T>
  concept remote_contiguous_range = std::ranges::random_access_range<T> &&
      remote_contiguous_iterator<std::ranges::iterator_t<T>> && requires(T t) {
    { t.rank() } -> std::convertible_to<std::size_t>;
  };


Requirements
""""""""""""

An object `t` of type ``T`` fulfills ``remote_contiguous_range`` if and only
if:

1. ``T`` fulfills ``std::ranges::random_access_range``.
2. ``T``'s iterator type fulfills ``remote_contiguous_iterator``.
3. ``t`` has a method ``rank`` returning the rank on which the range is
   located. For all iterators ``iter`` ``t.rank() == t.begin().rank()``.

Remarks
"""""""
All of the iterators in ``[begin(), end())`` should be contiguous iterators
with the same rank, and ``[begin().local(), end().local())`` should form a
contiguous range referencing the same memory, but locally. Not quite sure how
to express that concisely.

``distributed_contiguous_range``
================================

.. doxygenconcept:: dr::distributed_contiguous_range

A distributed contiguous range is a range consisting of multiple segments
distributed over multiple processes, where each each segment is a
remote contiguous range.

Defined in ``concepts.hpp``

::

  template <typename T>
  concept distributed_contiguous_range = std::ranges::random_access_range<T> &&
      requires(T t) {
    { t.segments() } -> std::ranges::random_access_range;
    {
      std::declval<std::ranges::range_value_t<decltype(t.segments())>>()
      } -> remote_contiguous_range;
  };


Requirements
""""""""""""

An object ``t`` of type ``T`` fulfills ``distributed_contiguous_range`` if and
only if:

1. ``T`` fulfills `std::ranges::random_access_range`
2. ``t`` has a method ``segments`` such that the ``t.segments()`` returns an
   ``std::ranges::random_access_range`` where each element is a
   ``remote_contiguous_range``.

Remarks
"""""""
Should there be other requirements, other than ``segments``?  Perhaps a
``distribution`` method to return an implementation-defined type describing the
distribution?
