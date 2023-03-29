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

   #include "mhp.hpp"

or::

   #include "shp.hpp"

No other includes. The names ``mhp.hpp`` and ``shp.hpp`` are likely to
change. In the future, we may support selective includes. Externally
exposed include paths cannot be changed without breaking compatibility.


Internal
========

Distributed ranges library developer.

Header files can be included in any order and therefore should include
their dependencies (internal and external). Include paths are always
relative to root (``-I`` path) and use ``<>``. Examples::

  #include <dr/views/transform.hpp>

Includes are protected with::

  #pragma once

Structure
---------

``dr``
  The top level ``dr`` directory protects against header name file
  collisions with other software.

``vendor``
  External header files that we distribute.

``dr/details``
  Does not fit elsewhere

``dr/shp``
  Single process, multi GPU model

``dr/mhp``
  Multi-process, single XPU model

``dr/views``
  Views shared between SHP/MHP

``dr/shp/algorithms``
  Algorithm implemenentations specific to SHP (e.g. ``shp::for_each``)

``dr/shp/containers``
  Container implemenentations specific to SHP
  (e.g. ``shp::distributed_vector``)

``dr/shp/views``
  Views implemenentations specific to SHP (e.g. ``shp::views::slice``)
