.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===============
 Include Files
===============

External
========

Application developers using distributed ranges library.

Use::

   #include "mp.hpp"

or::

   #include "sp.hpp"

No other includes. The names ``mp.hpp`` and ``sp.hpp`` are likely to
change. In the future, we may support selective includes. Externally
exposed include paths cannot be changed without breaking compatibility.


Internal
========

Distributed ranges library developer.

Header files can be included in any order and therefore should include
their dependencies (internal and external). Include paths are always
relative to root (``-I`` path) and use ``<>``. Example::

  #pragma once

  #include <algorithm>
  #include <ranges>

  #include <sycl/sycl.hpp>

  #include <oneapi/dpl/async>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/numeric>

  #include <dr/concepts/concepts.hpp>
  #include <dr/detail/onedpl_direct_iterator.hpp>
  #include <dr/sp/algorithms/execution_policy.hpp>
  #include <dr/sp/init.hpp>
  #include <dr/views/transform.hpp>

Use ``pragma once`` to protect against multiple inclusion. Start with
C++ headers as a block, blank line, external dependencies as 1 or more
blocks, blank line, internal dependencies as a block. ``clang-format``
sorts includes blocks that are not broken by blank lines.

Directory Structure
-------------------

``dr``
  The top level ``dr`` directory protects against header name file
  collisions with other software.

``vendor``
  External header files that we distribute.

``dr/detail``
  Does not fit elsewhere

``dr/sp``
  Single process, multi GPU model

``dr/mp``
  Multi-process, single XPU model

``dr/views``
  Views shared between SP/MP

``dr/sp/algorithms``
  Algorithm implemenentations specific to SP (e.g. ``sp::for_each``)

``dr/sp/containers``
  Container implemenentations specific to SP
  (e.g. ``sp::distributed_vector``)

``dr/sp/views``
  Views implemenentations specific to SP (e.g. ``sp::views::slice``)
