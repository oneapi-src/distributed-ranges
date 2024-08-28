// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// One-to-one with MKL out-of-place transpose

#include <chrono>
#include <iostream>

#include <stdlib.h>
#include <vector>

#include "mpi.h"

#include "mkl.h"

class TimeInterval {
public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

struct TransposeTimes {
  std::string name;
  double total;
  double local_transpose_1;
  double local_transpose_2;
  double block_exchange;
  double local_copy;

  TransposeTimes(std::string n) {
    name = n;
    total = 0;
    local_transpose_1 = 0;
    local_transpose_2 = 0;
    block_exchange = 0;
    local_copy = 0;
  };
};

void init(double *arr, int m, int n, int my_ID) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      arr[(i * n) + (j)] = ((i + my_ID * m) * n) + (j + 1);
}
void clear(double *arr, int m, int n) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      arr[(i * n) + (j)] = (double)(0.0);
}

void display(double *arr, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << arr[(i * n) + (j)] << " ";
    }
    std::cout << "\n";
  }
}

int main(int argc, char *argv[]) {

  int my_ID, Num_procs, phase;
  int Block_order, RowBlockSize, Block_size, col_start;
  int recv_from, send_to;

  MPI_Request recv_req, send_req;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <matrix_dim> <iterations>\n";
    exit(0);
  }
  int m, iters;
  m = atoi(argv[1]);
  if (m < Num_procs) {
    std::cout << "ERROR: matrix order " << m << " should at least # procs "
              << Num_procs << "\n";
    return 0;
  }

  iters = atoi(argv[2]);
  if (iters < 1) {
    std::cout << "ERROR: iterations must be >= 1 : " << iters << "\n";
    return 0;
  }

  Block_order = m / Num_procs;
  RowBlockSize = Block_order * m;
  Block_size = Block_order * Block_order;

  TransposeTimes cpu_times("cpu");

  if (my_ID == 0) {
    std::cout << "Matrix size: " << m << "x" << m << "\n";
    std::cout << "Matrix size per proc: " << Block_order << "x" << m << "\n";
    std::cout << "Num_procs: " << Num_procs << "\n";
    std::cout << "Iterations: " << iters << "\n";
  }
  double *A = (double *)mkl_malloc(RowBlockSize * sizeof(double), 64);
  double *B = (double *)mkl_malloc(RowBlockSize * sizeof(double), 64);

  double *send_buf = new double[Block_size];
  double *recv_buf = new double[Block_size];

  init(A, Block_order, m, my_ID);
  clear(B, m, Block_order);

  // for (int i=0; i<Num_procs; i++){
  //   if (my_ID == i){
  //     std::cout << "my_ID:"<<my_ID<<" After init A\n";
  //     display(A, Block_order, m);
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // std::cout << "\nExample of using mkl_dimatcopy transposition\n";

  TimeInterval mtimer;
  for (int i = 0; i < iters; ++i) {

    TimeInterval stepTimer;

    mkl_domatcopy('R', 'T', Block_order, Block_order, 1,
                  A + (my_ID * Block_order), m, (B + my_ID * Block_order), m);
    // for (int i=0; i<Num_procs; i++){
    //   if (my_ID == i){
    //     std::cout << "my_ID:"<<my_ID<<" After local transpose B\n";
    //     display(B, Block_order, m);
    //   }
    //   MPI_Barrier(MPI_COMM_WORLD);
    // }

    cpu_times.local_transpose_1 += stepTimer.Elapsed();

    for (phase = 1; phase < Num_procs; phase++) {
      send_to = (my_ID + phase) % Num_procs;
      recv_from = (my_ID - phase + Num_procs) % Num_procs;

      TimeInterval stepTimer2;

      mkl_domatcopy('R', 'T', Block_order, Block_order, 1,
                    A + (send_to * Block_order), m, send_buf, Block_order);

      cpu_times.local_transpose_2 += stepTimer2.Elapsed();

      // MPI_Barrier(MPI_COMM_WORLD);
      // MPI_Barrier(MPI_COMM_WORLD);
      // if (my_ID==0)
      //   std::cout << "Phase " << phase <<"\n";
      // for (int i=0; i<Num_procs; i++){
      //   if (my_ID == i){
      //     std::cout << "my_ID:"<<my_ID<<" After phase transpose recv
      //     send_buf: send_to: " << send_to <<"\n"; display(send_buf,
      //     Block_order, Block_order);
      //   }
      //   MPI_Barrier(MPI_COMM_WORLD);
      // }

      MPI_Irecv(recv_buf, Block_size, MPI_DOUBLE, recv_from, phase,
                MPI_COMM_WORLD, &recv_req);
      MPI_Isend(send_buf, Block_size, MPI_DOUBLE, send_to, phase,
                MPI_COMM_WORLD, &send_req);
      MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
      MPI_Wait(&send_req, MPI_STATUS_IGNORE);

      cpu_times.block_exchange += stepTimer2.Elapsed();

      // for (int i=0; i<Num_procs; i++){
      //   if (my_ID == i){
      //     std::cout << "my_ID:"<<my_ID<<" After phase recv recv_buf:
      //     recv_from: " << recv_from << "\n"; display(recv_buf, Block_order,
      //     Block_order);
      //   }
      //   MPI_Barrier(MPI_COMM_WORLD);
      // }

      col_start = recv_from * Block_order;
      // Copy from recv_buf to grid
      for (int i = 0; i < Block_order; ++i) {
        for (int j = 0; j < Block_order; ++j) {
          B[i * m + (j + col_start)] = recv_buf[i * Block_order + j];
        }
      }

      cpu_times.local_copy += stepTimer2.Elapsed();

      // for (int i=0; i<Num_procs; i++){
      //   if (my_ID == i){
      //     std::cout << "my_ID:"<<my_ID<<" After copy from recv_buf B\n";
      //     display(B, Block_order, m);
      //   }
      //   MPI_Barrier(MPI_COMM_WORLD);
      // }
    }
  }
  cpu_times.total += mtimer.Elapsed();

  // auto elapsed = mtimer.Elapsed();
  MPI_Barrier(MPI_COMM_WORLD);

  double total_local_transpose_time =
      cpu_times.local_transpose_1 + cpu_times.local_transpose_2;
  double total_block_transfer_time =
      cpu_times.block_exchange - cpu_times.local_transpose_2;
  double total_local_copy_time =
      cpu_times.local_copy - cpu_times.block_exchange;

  std::vector<double> total_times(Num_procs);
  std::vector<double> local_transpose_times(Num_procs);
  std::vector<double> block_transfer_times(Num_procs);
  std::vector<double> local_copy_times(Num_procs);

  MPI_Gather(&cpu_times.total, 1, MPI_DOUBLE, total_times.data(), 1, MPI_DOUBLE,
             0, MPI_COMM_WORLD);
  MPI_Gather(&total_local_transpose_time, 1, MPI_DOUBLE,
             local_transpose_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&total_block_transfer_time, 1, MPI_DOUBLE,
             block_transfer_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&total_local_copy_time, 1, MPI_DOUBLE, local_copy_times.data(), 1,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (my_ID == 0) {
    std::cout << "Total Time taken (s): " << cpu_times.total << "\n";
    std::cout << "Time per iteration (s): " << cpu_times.total / (double)iters
              << "\n";
    std::cout << "Bandwidth MB/s: "
              << (1.0e-6 * 2 * m * m * sizeof(double)) /
                     (cpu_times.total / (double)iters)
              << "\n\n\n";

    // std::cout << "Total times:\n";
    for (int i = 0; i < Num_procs; ++i)
      std::cout << total_times[i] << " ";
    std::cout << "| \n";

    // std::cout << "local_transpose_times:\n";
    for (int i = 0; i < Num_procs; ++i)
      std::cout << local_transpose_times[i] << " ";
    std::cout << "| \n";

    // std::cout << "block_transfer_times:\n";
    for (int i = 0; i < Num_procs; ++i)
      std::cout << block_transfer_times[i] << " ";
    std::cout << "| \n";

    // std::cout << "local_copy_times:\n";
    for (int i = 0; i < Num_procs; ++i)
      std::cout << local_copy_times[i] << " ";
    std::cout << "| \n";
  }

  MPI_Finalize();
  return 0;
}
