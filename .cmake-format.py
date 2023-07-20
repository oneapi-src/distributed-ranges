# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# Useful links about creating this config file:
# https://cmake-format.readthedocs.io/en/latest/configuration.html
# https://cmake-format.readthedocs.io/en/latest/configopts.html?highlight=layout_passes
# https://cmake-format.readthedocs.io/en/latest/format-algorithm.html

with section("format"):  # noqa: F821
    layout_passes = {"PargGroupNode": [(0, False)]}
