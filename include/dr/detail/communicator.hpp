// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr {

class communicator {
public:
  enum class tag {
    invalid,
    halo_forward,
    halo_reverse,
    halo_index,
  };

  communicator(MPI_Comm comm = MPI_COMM_WORLD) : mpi_comm_(comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    rank_ = rank;
    size_ = size;
  }

  auto size() const { return size_; }
  auto rank() const { return rank_; }
  auto prev() const { return (rank() + size() - 1) % size(); }
  auto next() const { return (rank() + 1) % size(); }
  auto first() const { return rank() == 0; }
  auto last() const { return rank() == size() - 1; }

  MPI_Comm mpi_comm() const { return mpi_comm_; }

  void barrier() const { MPI_Barrier(mpi_comm_); }

  void bcast(void *src, std::size_t count, std::size_t root) const {
    MPI_Bcast(src, count, MPI_BYTE, root, mpi_comm_);
  }

  void scatter(const void *src, void *dst, std::size_t count,
               std::size_t root) const {
    MPI_Scatter(src, count, MPI_BYTE, dst, count, MPI_BYTE, root, mpi_comm_);
  }

  template <typename T>
  void scatter(const std::span<T> src, T &dst, std::size_t root) const {
    assert(rng::size(src) >= size_);
    scatter(rng::data(src), &dst, sizeof(T), root);
  }

  void scatterv(const void *src, int *counts, int *offsets, void *dst,
                int dst_count, std::size_t root) const {
    assert(counts == nullptr || counts[rank()] == dst_count);
    MPI_Scatterv(src, counts, offsets, MPI_BYTE, dst, dst_count, MPI_BYTE, root,
                 mpi_comm_);
  }

  void gather(const void *src, void *dst, std::size_t count,
              std::size_t root) const {
    MPI_Gather(src, count, MPI_BYTE, dst, count, MPI_BYTE, root, mpi_comm_);
  }

  template <typename T>
  void gather(const T &src, std::span<T> dst, std::size_t root) const {
    assert(rng::size(dst) >= size_);
    gather(&src, rng::data(dst), sizeof(T), root);
  }

  template <typename T>
  void all_gather(const T *src, T *dst, std::size_t count) const {
    // Gather size elements from each rank
    MPI_Allgather(src, count * sizeof(T), MPI_BYTE, dst, count * sizeof(T),
                  MPI_BYTE, mpi_comm_);
  }

  template <typename T>
  void all_gather(const T &src, std::vector<T> &dst) const {
    assert(rng::size(dst) >= size_);
    all_gather(&src, rng::data(dst), 1);
  }

  template <typename T>
  void all_gather(std::vector<T> &src, std::vector<T> &dst) const {
    assert(rng::size(dst) >= size_ * rng::size(src));
    all_gather(rng::data(src), rng::data(dst), rng::size(src));
  }

  void gatherv(const void *src, int *counts, int *offsets, void *dst,
               std::size_t root) const {
    MPI_Gatherv(src, counts[rank()], MPI_BYTE, dst, counts, offsets, MPI_BYTE,
                root, mpi_comm_);
  }

  template <typename T>
  void isend(const T *data, std::size_t count, std::size_t dst_rank, tag t,
             MPI_Request *request) const {
    MPI_Isend(data, count * sizeof(T), MPI_BYTE, dst_rank, int(t), mpi_comm_,
              request);
  }

  template <rng::contiguous_range R>
  void isend(const R &data, std::size_t dst_rank, tag t,
             MPI_Request *request) const {
    isend(rng::data(data), rng::size(data), dst_rank, t, request);
  }

  template <typename T>
  void irecv(T *data, std::size_t size, std::size_t src_rank, tag t,
             MPI_Request *request) const {
    MPI_Irecv(data, size * sizeof(T), MPI_BYTE, src_rank, int(t), mpi_comm_,
              request);
  }

  template <rng::contiguous_range R>
  void irecv(R &data, std::size_t src_rank, tag t, MPI_Request *request) const {
    irecv(rng::data(data), rng::size(data), src_rank, t, request);
  }

  template <rng::contiguous_range R>
  void alltoall(const R &sendr, R &recvr, std::size_t count) {
    using T = typename R::value_type;
    MPI_Alltoall(rng::data(sendr), count * sizeof(T), MPI_BYTE,
                 rng::data(recvr), count * sizeof(T), MPI_BYTE, mpi_comm_);
  }

  template <typename T>
  void alltoallv(const T *sendbuf, std::vector<std::size_t> &sendcnt,
                 std::vector<std::size_t> &senddsp, T *recvbuf,
                 std::vector<std::size_t> &recvcnt,
                 std::vector<std::size_t> &recvdsp) {

    assert(rng::size(sendcnt) == size_);
    assert(rng::size(senddsp) == size_);
    assert(rng::size(recvcnt) == size_);
    assert(rng::size(recvdsp) == size_);

    std::vector<int> _sendcnt(size_);
    std::vector<int> _senddsp(size_);
    std::vector<int> _recvcnt(size_);
    std::vector<int> _recvdsp(size_);

    rng::transform(sendcnt, _sendcnt.begin(),
                   [](auto e) { return e * sizeof(T); });
    rng::transform(senddsp, _senddsp.begin(),
                   [](auto e) { return e * sizeof(T); });
    rng::transform(recvcnt, _recvcnt.begin(),
                   [](auto e) { return e * sizeof(T); });
    rng::transform(recvdsp, _recvdsp.begin(),
                   [](auto e) { return e * sizeof(T); });

    MPI_Alltoallv(sendbuf, rng::data(_sendcnt), rng::data(_senddsp), MPI_BYTE,
                  recvbuf, rng::data(_recvcnt), rng::data(_recvdsp), MPI_BYTE,
                  MPI_COMM_WORLD);
  }

  bool operator==(const communicator &other) const {
    return mpi_comm_ == other.mpi_comm_;
  }

private:
  MPI_Comm mpi_comm_;
  std::size_t rank_;
  std::size_t size_;
};

class rma_window {
public:
  void create(communicator comm, void *data, std::size_t size) {
    communicator_ = comm;
    drlog.debug("win create:: size: {}\n", size);
    MPI_Win_create(data, size, 1, MPI_INFO_NULL, comm.mpi_comm(), &win_);
  }

  void free() { MPI_Win_free(&win_); }

  bool operator==(const rma_window other) const noexcept {
    return this->win_ == other.win_;
  }

  void set_null() { win_ = MPI_WIN_NULL; }
  bool null() const noexcept { return win_ == MPI_WIN_NULL; }

  template <typename T> T get(std::size_t rank, std::size_t disp) const {
    T dst;
    get(&dst, sizeof(T), rank, disp * sizeof(T));
    return dst;
  }

  void get(void *dst, std::size_t size, std::size_t rank,
           std::size_t disp) const {
    drlog.debug("comm get:: ({}:{}:{})\n", rank, disp, size);
    MPI_Request request;
    MPI_Rget(dst, size, MPI_BYTE, rank, disp, size, MPI_BYTE, win_, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  void put(const auto &src, std::size_t rank, std::size_t disp) const {
    put(&src, sizeof(src), rank, disp * sizeof(src));
  }

  void put(const void *src, std::size_t size, std::size_t rank,
           std::size_t disp) const {
    drlog.debug("comm put:: ({}:{}:{})\n", rank, disp, size);
    MPI_Request request;
    MPI_Rput(src, size, MPI_BYTE, rank, disp, size, MPI_BYTE, win_, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  void fence() const { MPI_Win_fence(0, win_); }

  void flush(std::size_t rank) const {
    drlog.debug("flush:: rank: {}\n", rank);
    MPI_Win_flush(rank, win_);
  }

  const auto &communicator() const { return communicator_; }
  auto mpi_win() { return win_; }

private:
  dr::communicator communicator_;
  MPI_Win win_ = MPI_WIN_NULL;
};

} // namespace dr
