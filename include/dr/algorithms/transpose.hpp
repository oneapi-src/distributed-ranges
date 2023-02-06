// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <optional>

namespace lib {

template <typename T>
concept mdspan_2d = T::extents_type::rank() == 2;

template <typename T>
concept mdspan_pointer = std::is_same_v<typename T::container_type,
                                        std::vector<typename T::element_type>>;

template <typename T>
concept mdspan_row_major =
    std::is_same_v<typename T::layout_type, stdex::layout_right>;

template <typename T>
concept mdspan_col_major =
    std::is_same_v<typename T::layout_type, stdex::layout_left>;

template <typename T>
concept mdspan_regular = mdspan_pointer<T> &&
                         (mdspan_row_major<T> || mdspan_col_major<T>);

template <typename A, typename B> constexpr inline bool mdspan_same_layout() {
  return std::is_same_v<typename A::layout_type, typename B::layout_type>;
}

namespace collective {

template <typename T> inline auto mkl_layout() {
  if (mdspan_row_major<T>)
    return 'R';
  if (mdspan_col_major<T>)
    return 'C';
  assert(false);
}

template <typename src_type, typename dst_type>
inline void mkl_transpose(const src_type &src, dst_type &dst) {
  drlog.debug(nostd::source_location::current(),
              "MKL transpose: layout: {} rows: {} cols: {} lda: {} ldb: {}\n",
              mkl_layout<src_type>(), src.extents().extent(0),
              src.extents().extent(1), src.stride(0), dst.stride(0));
  mkl_domatcopy(mkl_layout<src_type>(), 'T', src.extents().extent(0),
                src.extents().extent(1), 1.0, src.data(), src.stride(0),
                dst.data(), dst.stride(0));
}

namespace {
// layout_right only arrays

template <typename T>
inline void transpose_local(size_t rows, size_t cols, const T *src,
                            size_t src_rows_distance, T *dst,
                            size_t dst_rows_distance) {
  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      dst[j * dst_rows_distance + i] = src[i * src_rows_distance + j];
    }
  }
}

template <>
inline void transpose_local(size_t rows, size_t cols, const float *src,
                            size_t src_rows_distance, float *dst,
                            size_t dst_rows_distance) {
  if (rows && cols)
    mkl_somatcopy('R', 'T', rows, cols, 1.0, src, src_rows_distance, dst,
                  dst_rows_distance);
}

template <>
inline void transpose_local(size_t rows, size_t cols, const double *src,
                            size_t src_rows_distance, double *dst,
                            size_t dst_rows_distance) {
  if (rows && cols)
    mkl_domatcopy('R', 'T', rows, cols, 1.0, src, src_rows_distance, dst,
                  dst_rows_distance);
}

template <>
inline void transpose_local(size_t rows, size_t cols, const MKL_Complex8 *src,
                            size_t src_rows_distance, MKL_Complex8 *dst,
                            size_t dst_rows_distance) {
  if (rows && cols)
    mkl_comatcopy('R', 'T', rows, cols, MKL_Complex8{1.0, 0.0}, src,
                  src_rows_distance, dst, dst_rows_distance);
}

template <>
inline void transpose_local(size_t rows, size_t cols, const MKL_Complex16 *src,
                            size_t src_rows_distance, MKL_Complex16 *dst,
                            size_t dst_rows_distance) {
  if (rows && cols)
    mkl_zomatcopy('R', 'T', rows, cols, MKL_Complex16{1.0, 0.0}, src,
                  src_rows_distance, dst, dst_rows_distance);
}

template <typename T, typename Extents> // layout_right only arrays
class transposer {
public:
  const size_t my_rank;
  const size_t num_proc;

private:
  const size_t rows;
  const size_t cols;
  const size_t sub_rows;
  const size_t sub_cols;

  std::array<std::vector<T>, 2> send_buf;
  std::array<size_t, 2> items_to_send;
  std::array<std::vector<T>, 2> receive_buf;

  const distributed_mdarray<T, Extents> &src_dmdarray;
  distributed_mdarray<T, Extents> &dst_mdarray;

  MPI_Request recv_req;
  MPI_Request send_req;

  inline size_t send_to_rank_idx(size_t proc_dst_forward) const {
    return (my_rank + proc_dst_forward) % num_proc;
  };
  inline int rcv_from_rank_idx(size_t proc_dst_backward) const {
    return (my_rank + num_proc - proc_dst_backward) % num_proc;
  };
  inline int buffer_for_idx(size_t proc_dst) const { return proc_dst % 2; };
  inline size_t sub_cols_of_idx(size_t idx) const {
    return cols < sub_cols * idx         ? 0
           : cols < sub_cols * (idx + 1) ? cols - sub_cols * idx
                                         : sub_cols;
  };
  inline size_t sub_rows_of_idx(size_t idx) const {
    return rows < sub_rows * idx         ? 0
           : rows < sub_rows * (idx + 1) ? rows - sub_rows * idx
                                         : sub_rows;
  };

  static T *raw_data(distributed_mdarray<T, Extents> &distributed_mdarray) {
    return distributed_mdarray.begin().container().local().data();
  }

  static const T *
  raw_data(const distributed_mdarray<T, Extents> &distributed_mdarray) {
    return distributed_mdarray.begin().container().local().data();
  }

  inline void compute_into(size_t sub_idx, T *dst,
                           size_t dst_rows_distance) const {
    drlog.debug("transposing idx:{}\n", sub_idx);
    transpose_local(
        sub_rows_of_idx(my_rank),
        sub_cols_of_idx(
            sub_idx), // watch that sometimes we send less columns than sub_cols
        raw_data(src_dmdarray) + sub_idx * sub_cols, cols, dst,
        dst_rows_distance);
  }

  // dimensions (rows, cols) of source submatrix BEFORE transpose (in original)
  std::pair<size_t, size_t> receive_items_extents(size_t proc_distance) const {
    return std::pair(sub_rows_of_idx(rcv_from_rank_idx(proc_distance)),
                     sub_cols_of_idx(my_rank));
  }

  std::pair<size_t, size_t> send_items_extents(size_t proc_distance) const {
    return std::pair(sub_rows_of_idx(my_rank),
                     sub_cols_of_idx(send_to_rank_idx(proc_distance)));
  }

  static size_t items_count(std::pair<size_t, size_t> extents) {
    return extents.first * extents.second;
  }

public:
  void compute_for_remote(size_t proc_distance) {
    assert(proc_distance > 0 &&
           proc_distance < num_proc); // proc_distance == 0 is my own proc
    compute_into(send_to_rank_idx(proc_distance),
                 send_buf[buffer_for_idx(proc_distance)].data(),
                 sub_rows_of_idx(my_rank));
  }

  void compute_for_me() {
    drlog.debug("computing my transposition, dstBase:{}\n",
                static_cast<void *>(raw_data(dst_mdarray)));
    compute_into(my_rank, raw_data(dst_mdarray) + my_rank * sub_rows, rows);
  }

  void send_receive_start(size_t proc_distance) {
    drlog.debug("SEND_RECEIVE_START dist:{}\n", proc_distance);
    const size_t receive_items_count =
        items_count(receive_items_extents(proc_distance));
    drlog.debug("receive {} items from {} proc start\n", receive_items_count,
                rcv_from_rank_idx(proc_distance));
    if (receive_items_count > 0) {
      src_dmdarray.comm().irecv(
          receive_buf[buffer_for_idx(proc_distance)].data(),
          receive_items_count, rcv_from_rank_idx(proc_distance),
          static_cast<communicator::tag>(proc_distance), &recv_req);
    }

    const size_t send_items_count =
        items_count(send_items_extents(proc_distance));
    drlog.debug("send {} items to {} proc start, buf: {}\n", send_items_count,
                send_to_rank_idx(proc_distance),
                send_buf[buffer_for_idx(proc_distance)]);
    if (send_items_count > 0)
      src_dmdarray.comm().isend(
          send_buf[buffer_for_idx(proc_distance)].data(), send_items_count,
          send_to_rank_idx(proc_distance),
          static_cast<communicator::tag>(proc_distance), &send_req);
  }

  void send_receive_wait(size_t proc_distance) {
    drlog.debug("WAIT dist:{}\n", proc_distance);
    if (items_count(receive_items_extents(proc_distance)))
      MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    if (items_count(send_items_extents(proc_distance)))
      MPI_Wait(&send_req, MPI_STATUS_IGNORE);
  }

  void dst_write(size_t proc_distance) {
    const std::pair<size_t, size_t> received_extents =
        receive_items_extents(proc_distance);

    drlog.debug("WRITE dist:{}, received rows:{} cols:{}, buf: {}\n",
                proc_distance, received_extents.first, received_extents.second,
                receive_buf[buffer_for_idx(proc_distance)]);

    if (items_count(received_extents) == 0)
      return;

    for (size_t dst_subrow_num = 0; dst_subrow_num < received_extents.second;
         ++dst_subrow_num) {
      memcpy(raw_data(dst_mdarray) + dst_subrow_num * rows +
                 sub_rows * rcv_from_rank_idx(proc_distance),
             receive_buf[buffer_for_idx(proc_distance)].data() +
                 received_extents.first * dst_subrow_num,
             received_extents.first * sizeof(T));
    }
  }

  transposer(distributed_mdarray<T, Extents> const &src,
             distributed_mdarray<T, Extents> &dst)
      : my_rank(src.comm().rank()), num_proc(src.comm().size()),
        rows(src.extents().extent(0)), cols(src.extents().extent(1)),
        sub_rows(partition_up(rows, num_proc)),
        sub_cols(partition_up(cols, num_proc)),
        send_buf{std::vector<T>(sub_rows_of_idx(my_rank) * sub_cols),
                 std::vector<T>(sub_rows_of_idx(my_rank) * sub_cols)},
        receive_buf{std::vector<T>(sub_rows * sub_cols_of_idx(my_rank)),
                    std::vector<T>(sub_rows * sub_cols_of_idx(my_rank))},
        src_dmdarray(src), dst_mdarray(dst) {
    drlog.debug(
        "created transposer, rows:{} cols:{}, sub_rows:{} sub_cols:{}\n", rows,
        cols, sub_rows, sub_cols);
    assert(dst_mdarray.extents().extent(0) == cols);
    assert(dst_mdarray.extents().extent(1) == rows);
  }
};

} // namespace

template <typename T, typename Extents> // layout_right only arrays
inline void transpose(const distributed_mdarray<T, Extents> &src,
                      distributed_mdarray<T, Extents> &dst) {
  static_assert(Extents::rank() == 2, "transpose is hard-coded for 2D layout");

  transposer<T, Extents> t{src, dst};

  if (t.num_proc == 1) {
    assert(t.my_rank == 0);
    t.compute_for_me();
  } else if (t.num_proc == 2) {
    t.compute_for_remote(1);
    t.send_receive_start(1);
    t.compute_for_me();
    t.send_receive_wait(1);
    t.dst_write(1);
  } else {
    t.compute_for_remote(1);
    t.send_receive_start(1);
    t.compute_for_remote(2);
    t.send_receive_wait(1);
    for (size_t d = 1; d < t.num_proc - 2; ++d) {
      t.send_receive_start(d + 1);
      t.dst_write(d);
      t.compute_for_remote(d + 2);
      t.send_receive_wait(d + 1);
    }
    t.send_receive_start(t.num_proc - 1);
    t.dst_write(t.num_proc - 2);
    t.compute_for_me();
    t.send_receive_wait(t.num_proc - 1);
    t.dst_write(t.num_proc - 1);
  }
}

template <typename T, typename SExtents,
          typename DExtents> // layout_right only arrays
inline void
transpose(int root, const distributed_mdarray<T, SExtents> &src,
          std::optional<std::experimental::mdspan<T, DExtents>> dst) {
  assert(dst.has_value() || root != src.comm().rank());
  std::vector<T> local_vec;

  if (src.comm().rank() == root) {
    assert(src.extents().extent(0) == dst.value().extent(1));
    assert(src.extents().extent(1) == dst.value().extent(0));

    size_t n = src.extents().extent(0) * src.extents().extent(1);
    local_vec.resize(n);
  }

  lib::copy(root, src.begin(), src.end(), local_vec.begin());

  if (src.comm().rank() == root) {
    transpose_local(src.extents().extent(0), src.extents().extent(1),
                    local_vec.data(), src.extents().extent(1),
                    dst.value().data_handle(), dst.value().extent(1));
  }
}

template <typename T, typename SExtents,
          typename DExtents> // layout_right only arrays
inline void transpose(int root,
                      std::optional<std::experimental::mdspan<T, SExtents>> src,
                      distributed_mdarray<T, DExtents> &dst) {
  assert(src.has_value() || root != dst.comm().rank());

  if (dst.comm().rank() == root) {
    assert(dst.extents().extent(0) == src.value().extent(1));
    assert(dst.extents().extent(1) == src.value().extent(0));

    std::vector<T> local_vec;
    size_t n = dst.extents().extent(0) * dst.extents().extent(1);
    local_vec.resize(n);
    transpose_local(src.value().extent(0), src.value().extent(1),
                    src.value().data_handle(), src.value().extent(1),
                    local_vec.data(), dst.extents().extent(1));
    lib::copy(root, local_vec.begin(), local_vec.end(), dst.begin());
  } else {
    lib::copy(root, nullptr, nullptr, dst.begin());
  }
}

template <typename T, typename T2 = std::nullopt_t,
          typename DExtents> // layout_right only arrays
inline void transpose(int root, T2 src, distributed_mdarray<T, DExtents> &dst) {
  transpose(root, std::optional<std::experimental::mdspan<T, DExtents>>(), dst);
}

template <typename T, typename T2 = std::nullopt_t,
          typename SExtents> // layout_right only arrays
inline void transpose(int root, const distributed_mdarray<T, SExtents> &src,
                      T2 dst) {
  transpose(root, src, std::optional<std::experimental::mdspan<T, SExtents>>());
}

template <mdspan_2d src_type, mdspan_2d dst_type>
inline void transpose(const src_type &src, dst_type &dst) {
  if constexpr (mdspan_regular<src_type> && mdspan_regular<dst_type> &&
                mdspan_same_layout<src_type, dst_type>()) {
    mkl_transpose(src, dst);
  } else {
    drlog.debug(nostd::source_location::current(), "Generic transpose\n");
    // Generic mdspan transpose
    for (std::size_t i = 0; i < src.extents().extent(0); i++) {
      for (std::size_t j = 0; j < src.extents().extent(1); j++) {
        dst(j, i) = src(i, j);
      }
    }
  }
}

} // namespace collective

} // namespace lib
