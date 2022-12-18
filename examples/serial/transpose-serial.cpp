#include "transpose-serial.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  transpose_serial<double> t;

  t.init(2, 6);
  t.compute();

  fmt::print("a: {}\n", t.a);
  fmt::print("b: {}\n", t.b);

  return 0;
}
