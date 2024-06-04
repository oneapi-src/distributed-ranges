.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===================
 Distributed Ranges
===================

.. image:: https://github.com/intel-sandbox/libraries.runtimes.hpc.dds.distributed-ranges/actions/workflows/pr.yml/badge.svg
   :target: https://github.com/intel-sandbox/libraries.runtimes.hpc.dds.distributed-ranges/actions/workflows/pr.yml
.. image:: https://www.bestpractices.dev/projects/8975/badge
   :target: https://www.bestpractices.dev/projects/8975

Productivity library for distributed and partitioned memory based on
C++ Ranges.

About
-----

Distributed Ranges is a productivity library for distributed and partitioned memory based on C++ ranges.
It offers a collection of data structures, views, and algorithms for building generic abstractions
and provides interoperability with MPI, SHMEM, SYCL and OpenMP and portability on CPUs and GPUs.
NUMA-aware allocators and distributed data structures facilitate development of C++ applications
on heterogeneous nodes with multiple devices and achieve excellent performance and parallel scalability
by exploiting local compute and data access.

Documentation
-------------

- Usage:
  - Introductory presentation: `Distributed Ranges, why you need it`_, 2024
  - Article: `Get Started with Distributed Ranges`_, 2023
  - Tutorial: `Sample repository showing Distributed Ranges usage`_
- Design / Implementation:
  - Conference paper: `Distributed Ranges, A Model for Distributed Data Structures, Algorithms, and Views`_, 2024
  - Talk: `CppCon 2023; Benjamin Brock; Distributed Ranges`_, 2023
  - Technical presentation: `Intel Innovation'23`_, 2023
  - `API specification`_
  - `Doxygen`_

Contact us
----------

We seek collaboration opportunities and welcome feedback on ways to extend the library,
according to developer needs. Contact us by writing a `new issue`_.


Examples
--------

See `Sample repository showing Distributed Ranges usage`_ for a few well explained examples.
Additionally you may build all tests of this repository to see and run much more examples.

Build and test with gcc for CPU::

  CXX=g++-12 cmake -B build
  make -C build -j all test

Build and test with ipcx for SYCL && CPU/GPU::

  CXX=icpx cmake -B build -DENABLE_SYCL=ON

See how example is run and the output::

  cd build
  ctest -VV

Adding Distributed Ranges to your project
-----------------------------------------

See `Sample repository showing Distributed Ranges usage`_
for a live example how to write CMakeLists.txt. Alternatively you may read details below.

If your project uses CMAKE, add the following to your
``CMakeLists.txt`` to download the library::

  find_package(MPI REQUIRED)
  include(FetchContent)
  FetchContent_Declare(
    dr
    GIT_REPOSITORY https://github.com/oneapi-src/distributed-ranges.git
    GIT_TAG main
    )
  FetchContent_MakeAvailable(dr)

The above will define targets that can be included in your project::

  target_link_libraries(<application> MPI::MPI_CXX DR::mpi)

If your project does not use CMAKE, then you need to download the
library, and install it into a prefix::

  git clone https://github.com/oneapi-src/distributed-ranges.git dr
  cd dr
  cmake -B build -DCMAKE_INSTALL_PREFIX=<prefix>
  make -C build install
  cmake -B build-fmt -DCMAKE_INSTALL_PREFIX=<prefix> build/_deps/cpp-format-src
  make -C build-fmt install

Use ``-I`` and ``-L`` to find headers and libs during compilation::

  g++ -std=c=++20 -I <prefix>/include -L <prefix>/lib -L /opt/intel/oneapi/mpi/latest/lib/release -lfmt -lmpicxx -lmpi

Logging
-------

Add this to your main to enable logging::

  std::ofstream logfile(fmt::format("dr.{}.log", comm_rank));
  dr::drlog.set_file(logfile);


Contributing
------------

See [CONTRIBUTING](./CONTRIBUTING.md)


See also
--------

`Fuzz Testing`_
  Fuzz testing of distributed ranges APIs

`Spec Editing`_
  Editing the API document

`Print Type`_
  Print types at compile time:

`Testing`_
  Test system maintenance

`Security`_
  Security policy

.. _`Security`: SECURITY.md
.. _`Testing`: doc/developer/testing
.. _`Spec Editing`: doc/spec/README.rst
.. _`Fuzz Testing`: test/fuzz/README.rst
.. _`Print Type`: https://stackoverflow.com/a/14617848/2525421

.. _`Distributed Ranges, why you need it`: https://github.com/oneapi-src/distributed-ranges/blob/main/doc/presentations/Distributed%20Ranges%2C%20why%20you%20need%20it.pdf
.. _`Get Started with Distributed Ranges`: https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-distributed-ranges.html
.. _`Sample repository showing Distributed Ranges usage`: https://github.com/oneapi-src/distributed-ranges-tutorial
.. _`Distributed Ranges, A Model for Distributed Data Structures, Algorithms, and Views`: https://dl.acm.org/doi/10.1145/3650200.3656632
.. _`CppCon 2023; Benjamin Brock; Distributed Ranges`: https://www.youtube.com/watch?v=X_dlJcV21YI
.. _`Intel Innovation'23`: https://github.com/oneapi-src/distributed-ranges/blob/main/doc/presentations/Distributed%20Ranges.pdf
.. _`API specification`: https://oneapi-src.github.io/distributed-ranges/spec/
.. _`Doxygen`: https://oneapi-src.github.io/distributed-ranges/doxygen/
.. _`new issue`: issues/new
