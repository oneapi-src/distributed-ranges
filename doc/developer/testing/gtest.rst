.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

=============
 Google Test
=============

We use Google Test for unit tests. They are located at

``test/gtest/mhp``
  tests for mhp programming model
``test/gtest/shp``
  tests for shp programming model
``test/gtest/include``
  testing code shared between shp/mhp
``test/gtest/include/common``
  tests shared between shp/mhp.

Google test creates a single binary that can run all tests in a single
invocation. We have ``shp-tests`` and ``mhp-tests`` for the 2 models.
``mhp-tests`` relies on MPI and the binary is executed with varying
numbers of processes. ``shp-tests`` is executed with varying number of
SYCL devices.

Common Tests
============

When appropriate, we use the same test for shp/mhp for test writing
productivity and to ensure compatibility between the 2 models. Common
tests are found in ``test/gtest/include/common``, and included in
``shp-tests.cpp`` and ``mhp-tests.cpp``.

There are intentional differences between the shp/mhp that make it
impossible to directly share tests. We have some support to cover
small common differences. Use your judgement to decide if there should
be unified or separate tests.

We should converge on common practices for writing shared tests, but
it is too early to mandate specific practices. This is what I have
done that may be useful.

Shims
-----

Shims are generic names/functions that are defined differently in mhp
and shp.

* Use ``xhp::`` as the namespace when you need ``shp::`` or ``mhp::``
  namespaces.
* ``default_policy(dv)`` where ``dv`` is a distributed vector. Use for
  algorithms that require a policy.
* Use ``barrier()`` when mhp requires a barrier. It does nothing for
  shp. A barrier is typically needed when you have a test that checks
  some values, and then modifies the results. The barrier ensures that
  the check is finished before any rank modifies the test. Avoid the
  need for barriers by breaking into multiple tests.
* Use ``fence()`` when mhp requires a fence. It does nothing for
  shp. A fence is needed when there is a transition between local and
  global updates. Mhp collectives will automatically do a fence when
  needed. Fences are needed when using distributed vector iterators or
  indexing.

Templated Tests
---------------

Templated tests are used to instantiate the same test with different
types which is useful for covering different data types and
allocators.

``AllTypes``
  Includes a variety of ``distributed_vector`` types that can be used
  with a ``TYPED_TEST_SUITE``. In a typed test suite ``TypeParam`` is
  the ``distributed_vector`` type being tested in the instantiation of
  the test.
``Ops1``, ``Ops2``, ``Ops3``
  Provides some boiler plate code to create and initialize
  ``distributed_vector`` and ``std::vector``. Provides inputs for 1-3
  operands and initializes them with ``iota``.

Reference Results
-----------------

I usually compare distributed ranges views/algorithms against the
``std::`` equivalents when applied to ``std::vector``. It an be
quicker to write tests, and it reduces the chance that there is a
misunderstanding about the behavior of ``std::``.

Consistency Checking
--------------------

The function ``check_view`` is a utility function to test that the
distributed ranges and ``std::`` view are the same, and does
consistency checks on the segments in a view.  ``check_mutate_view``
does everything that ``check_view`` does and also verifies that the
view can be written.
