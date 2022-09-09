#include <iostream>
#include <vector>

#include "common.hpp"

//#include "distributed-ranges.hpp"

void local() {
  const size_t n = 10;
  std::vector<int> a(n), b(n), c(n), c_ref(n);

  // Initialize
  set_step(a, 0);
  set_step(b, 10);
  set_step(c, 0, 0);

  // Want to define abstract index space and then iterate over it. For
  // remote, I specify how the index space is decomposed and then I
  // can use that to control data distribution and owner computes. Is
  // that built into execution policy?
  std::vector<int> vrange(n);
  set_step(vrange, 0);

  for (auto i : vrange) {
    c[i] = a[i] + b[i];
  }

  show("a:", a);
  show("b:", b);
  show("c:", c);

  set_step(c_ref, 10, 2);
  assert(check(c, c_ref) == 0);
}

void remote() {

  // Should be similar in style to local, but handle remote data
}

int main(int argc, char *argv[]) {

  local();
  remote();

  return 0;
}
