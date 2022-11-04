#include "transpose-serial.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  transpose_serial<double> t;

  t.init(2, 6);
  t.compute();

  show("a: ", t.a);
  show("b: ", t.b);

  return 0;
}
