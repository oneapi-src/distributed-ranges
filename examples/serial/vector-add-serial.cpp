#include "vector-add-serial.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  vector_add_serial<int> adder;

  adder.init(10);
  adder.compute();

  show("a: ", adder.a);
  show("b: ", adder.b);
  show("c: ", adder.c);

  return 0;
}
