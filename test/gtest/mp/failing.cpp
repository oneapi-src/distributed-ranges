#include <dr/mp.hpp>
#include "mpi.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  int size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

#ifdef SYCL_LANGUAGE_VERSION
  sycl::queue q = dr::mp::select_queue();
  fmt::print("Running on sycl device: {}, memory: device\n", q.get_device().get_info<sycl::info::device::name>());
  dr::mp::init(q, sycl::usm::alloc::device);
#else
  assert(false)
#endif

  auto dist = dr::mp::distribution().halo(1);
  dr::mp::distributed_vector<int> array(10, dist);
 fill(array, 7);

 if (rank == 0) {
   for (int i = 0; i <= 4; i++) {
     std::cout << i << ": " << (array.segments()[0].begin() + i).get() << "\n";
   }
   std::cout << 5 << ": " << (array.segments()[1].begin() + 1).get() << "\n";
   for (int i = 0; i <= 4; i++) {
     (array.segments()[0].begin() + i).put(i);
   }
   (array.segments()[1].begin() + 1).put(5);
   for (int i = 0; i <= 4; i++) {
     std::cout << i << ": " << (array.segments()[0].begin() + i).get() << "\n";
   }
   std::cout << 5 << ": " << (array.segments()[1].begin() + 1).get() << "\n";
 } else {
   std::cout << 4 << ": " << (array.segments()[0].begin() + 4).get() << "\n";
   for (int i = 0; i <= 4; i++) {
     std::cout << i + 5 << ": " << (array.segments()[1].begin() + i).get() << "\n";
   }
   (array.segments()[0].begin() + 4).put(5);
   for (int i = 0; i <= 4; i++) {
     (array.segments()[1].begin() + i).put(i + 5);
   }
   std::cout << 4 << ": " << (array.segments()[0].begin() + 4).get() << "\n";
   for (int i = 0; i <= 4; i++) {
     std::cout << i + 5 << ": " << (array.segments()[1].begin() + i).get() << "\n";
   }
 }

//  array.fence();

  std::cout << "Begin halo\n";
  array.halo().exchange();
  std::cout << "End halo\n";

//  array.fence();

 if (rank == 0) {
   for (int i = 0; i <= 4; i++) {
     std::cout << i << ": " << (array.segments()[0].begin() + i).get() << "\n";
   }
   std::cout << 5 << ": " << (array.segments()[1].begin() + 1).get() << "\n";
 } else {
   std::cout << 4 << ": " << (array.segments()[0].begin() + 4).get() << "\n";
   for (int i = 0; i <= 4; i++) {
     std::cout << i + 5 << ": " << (array.segments()[1].begin() + i).get() << "\n";
   }
 }
}