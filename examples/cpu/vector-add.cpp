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

// Here, I assume a SPMD-style execution.
// Every process runs every line of code, and I can
// access a proc's rank number with `lib::rank()` and
// the total number of procs with `lib::nprocs()`.
// Here, I allocate the remote_vectors on rank 0,
// then create remote_spans, which are broadcast.
// Finally, the compute is performed on rank 1.
// (You likely wouldn't actually want to do this,
//  but this demonstrates the API I have in mind.)
void remote() {
  const size_t n = 10;

  lib::remote_span<int> a_span, b_span, c_span, c_ref_span;

  // Allocate remote vectors on rank 0.
  if (lib::rank() == 0) {
    pml::remote_vector<int> a(n), b(n), c(n), c_ref(n);
    a_span = a;
    b_span = b;
    c_span = c;
    c_ref_span = c_ref;

    // Assign them some data locally
    set_step(a, 0);
    set_step(b, 10);
    set_step(c, 0, 0);
  }

  a_span = lib::broadcast(0, a_span);
  b_span = lib::broadcast(0, b_span);
  c_span = lib::broadcast(0, c_span);
  c_ref_span = lib::broadcast(0, c_ref_span);

  // Now do the computation on rank 1, using the remote spans.
  if (lib::rank() == 1) {

    for (size_t i = 0; i < a_span.size(); i++) {
      c_span[i] = a_span[i] + b_span[i];
    }
  }

  // Ensure computation has finished and data has been updated.
  lib::barrier();

  // Check data on rank 0.
  if (lib::rank() == 0) {
    show("a:", a);
    show("b:", b);
    show("c:", c);

    set_step(c_ref, 10, 2);
    assert(check(c, c_ref) == 0);
  }
}

// Similar to `remote`, here I envision using a SPMD style of
// execution.  However, we use distributed objects instead of
// remote ones, and so we call the `distributed_vector`
// constructors collectively.
void distributed() {
  const size_t n = 10;

  // Construct the distributed vectors collectively.
  // I envision `distributed_vector` having a very similar
  // interface to `distributed_span`, except it owns its own
  // data.  It should have an optional `distribution` parameter
  // that analogous to the `accessor` parameter we talked about
  // for `distributed_span`.
  lib::distributed_vector<int> a(n), b(n), c(n), c_ref(c);

  // Initialize using the distributed `lib::for_each` (look at function below.)
  d_set_step(a, 0);
  d_set_step(b, 10);
  d_set_step(c, 0, 0);

  // For the iteration, there are two ways I imagine you could do it.
  // The first one involves 'zipping' the distributed spans together.

  auto zipped_arrays = lib::zip_view(c, a, b)

      // `zipped_arrays` is now a distributed range.  It represents the three
      // distributed arrays, c, a, and b, with all of their elements zipped
      // together. (Each element of `zipped_arrays` contains a tuple with three
      // references, each
      //  to the corresponding element of c, a, and b.)

      lib::for_each(lib::parallel_locality, zipped_arrays,
                    [=](auto &&c_ref, auto &&a_ref, auto &&b_ref) {
                      c_ref = a_ref + b_ref;
                    });

  // Now, there's a question about exactly what `parallel_locality` means here,
  // since we haven't defined the locality of `zipped_arrays` explicitly.  We
  // just zipped three distributed arrays, which happened to have the same
  // locality.  They didn't have to, though (e.g. one of them could have been
  // block cyclic). I suppose a reasonable choice would be to take the locality
  // of the first range, but that should probably be handled as an argument to
  // `lib::zip_view` if it's configurable.

  /*
    // Here's an alternate implementation using an index-based iteration rather
    than zip.
    // I'm just using raw `iota_view` here, but I would imagine you might add
    sycl-like
    // range and ndrange types for convenience.

    // The key here is (for the implementer) having a way to specify an explicit
    distribution
    // over the iteration space.  It would be nice if the accessor in the
    distributed_span
    // (and possibly other distributed ranges) were accessible, so you could
    pass `a.get_accessor()`
    // or `a.get_distribution()` into your execution policy to guarantee the
    same distribution
    // as `a`.
    // Here, I'm just imagining that `parallel_explicit` would by default use a
    normal block distribution,
    // which happens to be the same distribution as a, b, and c.
    lib::for_each(lib::parallel_explicit(), std::views::iota_view(0, c.size()),
                  [&](int i) {
                    c[i] = a[i] + b[i];
                  });
  */

  // See d_show below. Similar to d_set_step, except I imagine you'd want to do
  // it on one process so the lines print out in order.
  d_show("a:" a);
  d_show("b:" b);
  d_show("c:" c);
}

// A `distributed_vector<int>` can be automatically converted
// into a `distributed_span<int>` (and so might some other types).
void d_set_step(lib::distributed_span<int> span, int value) {
  // Call the collective function `for_each` with the `lib::parallel_locality`
  // policy, which should be the default (or whatever we end up calling it).
  // It should do the default thing of having each process manipulate the parts
  // of the range that it owns.
  lib::for_each(lib::parallel_locality, span, [=](int &v) { v = value; });

  /*
    // This is what I imagine for_each doing:
    for (auto&& subspan : span.subspans()) {
      if (subspan.rank() == lib::rank()) {
        std::for_each(subspan.begin().local(), subspan.end().local(),
                      [=](int& v) {
                        v = value;
                      });
      }
    }
  */
}

void d_show(std::string label, lib::distributed_span<int> span) {
  std::cout << label << ": ";
  lib::for_each(lib::sequential, span, [](auto &&ref) { std::cout << ref; })
          std::cout
      << std::endl;
}

int main(int argc, char *argv[]) {

  local();
  remote();

  return 0;
}
