// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "vector-add-serial.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  vector_add_serial<int> adder;

  adder.init(10);
  adder.compute();

  fmt::print("a: {}\n", adder.a);
  fmt::print("b: {}\n", adder.b);
  fmt::print("c: {}\n", adder.c);

  return 0;
}
