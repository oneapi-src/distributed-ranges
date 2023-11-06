.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

============
 Release 0.1
============

#. Library

   #. Backends

      #. MPI + SYCL + shared/device memory
      #. iSHMEM + SYCL backend (subset of API)

   #. Distributed containers:

      #. ``distributed_vector``

         #. models distributed range
         #. distribution:

            #. block/interleave
            #. halo

      #. ``distributed_mdarray``

         #. models distributed range
         #. distribution:

            #. block/interleave on leading dimension
            #. halo

   #. Disributed Range Algorithms

      #. ``for_each``
      #. ``for_each_tile``
      #. ``copy``
      #. ``fill``
      #. ``reduce``
      #. ``sort``
      #. ``inclusive_scan``
      #. ``exclusive_scan``
      #. ``transform``

   #. Distributed Range Views

      #. ``zip``
      #. ``iota``
      #. ``take``
      #. ``drop``
      #. ``counted``
      #. ``enumerate``
      #. ``transform``

   #. Distributed Mdspan Algorithms

      #. ``mdfor_each``
      #. ``mdfor_each_stencil``
      #. ``mdfor_each_tile``

      Commit to current API.

   #. Distributed Mdspan Views

      #. ``mdsubspan``

   #. Communication primitives

      #. ``copy`` to/from local and DR
      #. distributed range remote reference exposed as [index]
      #. distributed mdspan remote reference exposed as (index0, index1, ...)
      #. halo exchange

#. Documentation

   #. API reference
   #. Getting started
   #. Tutorial

#. Examples

   #. Vector Add
   #. Vector add with views
   #. Dot product
   #. 2d Stencil

#. Tests

   #. Unit testing for every container, view, algorithm

#. Benchmarks

   #. Algorithms

     #. Streams
     #. Black Scholes
     #. Reduce
     #. Wave Equation

   #. Methodology

      #. Strong scaling: largest size that fits on 1 GPU, scale GPUs
      #. Weak scaling: largest size that fits on 1 GPU, scale size with GPUs

#. Performance Goal

   #. 8 Aurora nodes (6 cards with 2 tiles), weak scaling
   #. 80% of perfect speedup for single node reference code or
      roofline analysis
