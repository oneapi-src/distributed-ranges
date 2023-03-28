.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===============
 Include Style
===============

External
========

Use::

   #include "mhp.hpp"

or::

   #include "shp.hpp"

No other includes. ``MHP/SHP`` are likely to change. In the future, we
can support selective includes with a commitment to compatibility.

Internal
========

Header files can be included in any order and therefore include should
their dependencies (internal and external). Include paths are always
relative to root and using ``<>``. Includes are protected with::

  #pragma once
