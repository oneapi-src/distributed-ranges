.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

=======
 CTest
=======

We use ``ctest``, which is part of ``cmake`` as a top-level test
runner. It runs google test and examples. To invoke tests::

  ctest

or::

  make test

To see more output, do::

  ctest -VV
