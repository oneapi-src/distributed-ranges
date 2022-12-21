.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

============
 Containers
============

*Containers* own storage.

The storage for a *distributed container* is divided over multiple
processes and can be accessed by any process in the team.

.. toctree::
   :maxdepth: 1

   distributed_mdarray
   distributed_vector

The storage for a *remote container* resides in a single process and
can be accessed by any process in the team.

.. toctree::
   :maxdepth: 1

   remote_vector
