.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

=============
 Using Vtune
=============

WSL
===

To prepare, compile the program with optimization and ``-g``.

The simplest option is to run ``vtune-gui`` inside a Linux VM on
WSL. Create a new project for the application, configure the
arguments, select *hot spot* with *hardware sampling*.

I tried some other options. I am documenting here in case they are
needed.

Command line collection and vtune-gui inside Linux VM.  Generate a
profile in a directory called ``vtune``::

  rm -rf vtune && vtune -collect hotspots -knob sampling-mode=hw -result-dir vtune ./mhp-bench --benchmark_filter=^Mdspan_ --reps 10

View it in VTune::

  vtune-gui vtune

If you can collect the profile on Linux but cannot get the GUI to work
from Linux, you can run the GUI on Windows.  There are separate file
systems for linux and windows with WSL. Cross OS accesses are very
slow. Generate the profile on Linux using the Linux filesystem, copy
to the windows filesystem, and launch the GUI from windows. In the
following example, ``~/windows`` is a symlink from linux to my Windows
home directory. I put the profile in a directory called
``vtune``. Collect a profile::

  rm -rf vtune ~/windows/Downloads/vtune && vtune -collect hotspots -knob sampling-mode=hw -result-dir vtune ./mhp-bench --benchmark_filter=^Mdspan_ --reps 10 && cp -r vtune ~/windows/Downloads/

View in GUI::

  "C:\Program Files (x86)\intel\oneAPI\vtune\latest\bin64\vtune-gui.exe" vtune


Remote Profiling
================

Follow the directions above, except that you need to ``scp`` the
profile, source files, and binary from Linux to Windows.

Examining Results
=================

When looking at ``mhp-bench``, I start with the *Top-down Tree*
tab. Click on *Total*. Double click on the benchmark you want to
examine. Click on *Assembly* to see source and assembly code.

If you are not collecting/viewing on the same system, you may have to
give the path of the source file and binary. If you are using WSL to
collect on Linux and view on Windows, WSL will let you access the
Linux files from the *Open File* box. I see the Linux filesystem in
the explorer in the bottom of the left pane under *Linux*. Right
clicking on a directory on Windows lets you *Pin to Quick Access* so
you can find it quickly in the left pane.
