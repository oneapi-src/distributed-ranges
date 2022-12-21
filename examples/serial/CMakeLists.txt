# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(vector-add-serial vector-add-serial.cpp)
target_link_libraries(vector-add-serial PRIVATE fmt::fmt)
add_test(vector-add-serial ./vector-add-serial)

add_executable(transpose-serial transpose-serial.cpp)
target_link_libraries(transpose-serial PRIVATE MKL::MKL range-v3 fmt::fmt)
add_test(transpose-serial ./transpose-serial)
