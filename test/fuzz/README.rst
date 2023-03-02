.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===========
 Fuzz Test
===========

Build the test::

  CXX=clang++ cmake -B build
  cd build/fuzz/cpu
  make -j
  ./cpu-fuzz -max_len=16

The command asserts when it finds an error. Otherwise it runs forever
so kill it to stop testing. When it finds an error, it writes the
input to a file in the current directory. To run again for just that
input::

  ./cpu-fuzz . .
