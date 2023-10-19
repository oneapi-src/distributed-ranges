.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

=========
Placement
=========

Proper placement of processes and SYCL devices on physical resources
is essential for best utilization and well controlled experiments. For
a multi-process distributed ranges program, each process uses 1 thread
or 1 SYCL device for computation. Create enough processes to fully utilize
the desired number of sockets and GPUs.  Each process should be bound to a
dedicated core and a SYCL device should be bound to a dedicated tile
or a dedicated CPU socket.

For a single process distributed ranges program, all available compute
units (hardware threads for CPU, EU's for GPU) in the SYCL devices are
used for computation. Each SYCL device should be bound to a dedicated
tile or dedicated CPU socket.

For flexibility and ease of development, programs should rely on
external mechanisms to control placement. MPI can control the
placement of processes to cores/sockets and binding a specific GPU to
a process. See `CPU Pinning`_ and `GPU Pinning`_ for more
information. SYCL runtime can control the mapping of SYCL devices to
GPU or CPU, and the number of devices available. See
`ONEAPI_DEVICE_SELECTOR`_ for more information. When using a SYCL CPU
device, the OpenMP runtime can control the mapping of threads to
cores.

.. _`ONEAPI_DEVICE_SELECTOR`: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector
.. _`GPU Pinning`: https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-8/gpu-pinning.html
.. _`CPU Pinning`: https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-pinning-simulator.html#gs.10glno

MHP/CPU
=======

The examples that follow are for a 2 socket system with 24 cores in
each socket.  Fully utilize 1 socket::

  I_MPI_PIN_DOMAIN=core I_MPI_PIN_ORDER=compact I_MPI_PIN_CELL=unit mpirun -n 24 ./mhp-bench

Fully utilize 2 sockets::

  I_MPI_PIN_DOMAIN=core I_MPI_PIN_ORDER=compact I_MPI_PIN_CELL=unit mpirun -n 48 ./mhp-bench

MHP/SYCL Using CPU Devices
==========================

The examples that follow are for a 2 socket system with 24 cores in
each socket.  Use ``sycl-ls`` to discover available devices::

  (cvenv) rscohn1@anpfclxlin02:shp$ sycl-ls
  [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.15.3.0.20_160000]
  [opencl:cpu:1] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8260M CPU @ 2.40GHz 3.0 [2023.15.3.0.20_160000]
  (cvenv) rscohn1@anpfclxlin02:shp$

Fully utilize 1 socket with 1 SYCL device::

  ONEAPI_DEVICE_SELECTOR=opencl:cpu I_MPI_PIN_DOMAIN=socket mpirun -n 1 ./mhp-bench --sycl

Fully utilize 2 sockets with 2 SYCL devices::

  ONEAPI_DEVICE_SELECTOR=opencl:cpu I_MPI_PIN_DOMAIN=socket I_MPI_PIN_ORDER=compact I_MPI_PIN_CELL=unit mpirun -n 2 ./mhp-bench --sycl

Fully utilize all sockets with 1 SYCL device. Programs that use a
single SYCL device to use 2 sockets typically have poor NUMA behavior
so it is recommended to instead use 1 device per socket.::

  ONEAPI_DEVICE_SELECTOR=opencl:1 mpirun -n 1 ./mhp-bench --sycl

SHP Using CPU Devices
=====================

The examples that follow are for a 2 socket system with 24 cores in
each socket.  Use ``sycl-ls`` to discover available devices::

  (cvenv) rscohn1@anpfclxlin02:shp$ sycl-ls
  [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.15.3.0.20_160000]
  [opencl:cpu:1] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8260M CPU @ 2.40GHz 3.0 [2023.15.3.0.20_160000]
  (cvenv) rscohn1@anpfclxlin02:shp$

Fully utilize 1 socket with 1 SYCL device::

  ONEAPI_DEVICE_SELECTOR=opencl:1.0 KMP_AFFINITY=compact ./shp-bench

Fully utilize 2 sockets with 2 SYCL devices::

  ONEAPI_DEVICE_SELECTOR=opencl:1.0,1.1 KMP_AFFINITY=compact ./shp-bench
