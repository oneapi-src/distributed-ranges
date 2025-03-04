// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/allocator.hpp>
#include <dr/mp/containers/distribution.hpp>
#include <dr/mp/containers/dual_segment.hpp>

namespace dr::mp {

static constexpr std::size_t DUAL_SEGMENTS_PER_PROC = 2;

class DualMpiBackend {
  dr::rma_window win_;

public:
  void *allocate(std::size_t data_size) {
    assert(data_size > 0);
    void *data = __detail::allocator<std::byte>().allocate(data_size);
    DRLOG("called MPI allocate({}) -> got:{}", data_size, data);
    win_.create(default_comm(), data, data_size);
    active_wins().insert(win_.mpi_win());
    return data;
  }

  void deallocate(void *data, std::size_t data_size) {
    assert(data_size > 0);
    DRLOG("calling MPI deallocate ({}, data_size:{})", data, data_size);
    active_wins().erase(win_.mpi_win());
    win_.free();
    __detail::allocator<std::byte>().deallocate(static_cast<std::byte *>(data),
                                                data_size);
  }

  void getmem(void *dst, std::size_t offset, std::size_t datalen,
              int segment_index) {
    const std::size_t peer = get_peer(segment_index);

    DRLOG("calling MPI get(dst:{}, "
          "segm_offset:{}, size:{}, peer:{})",
          dst, offset, datalen, peer);

#if (MPI_VERSION >= 4) ||                                                      \
    (defined(I_MPI_NUMVERSION) && (I_MPI_NUMVERSION > 20211200000))
    // 64-bit API inside
    win_.get(dst, datalen, peer, offset);
#else
    for (std::size_t remainder = datalen, off = 0UL; remainder > 0;) {
      std::size_t s = std::min(remainder, (std::size_t)INT_MAX);
      DRLOG("{}:{} win_.get total {} now {} bytes at off {}, dst offset {}",
            default_comm().rank(), __LINE__, datalen, s, off, offset + off);
      win_.get((uint8_t *)dst + off, s, peer, offset + off);
      off += s;
      remainder -= s;
    }
#endif
  }

  void putmem(void const *src, std::size_t offset, std::size_t datalen,
              int segment_index) {
    const std::size_t peer = get_peer(segment_index);

    DRLOG("calling MPI put(segm_offset:{}, "
          "src:{}, size:{}, peer:{})",
          offset, src, datalen, peer);

#if (MPI_VERSION >= 4) ||                                                      \
    (defined(I_MPI_NUMVERSION) && (I_MPI_NUMVERSION > 20211200000))
    // 64-bit API inside
    win_.put(src, datalen, peer, offset);
#else
    for (std::size_t remainder = datalen, off = 0UL; remainder > 0;) {
      std::size_t s = std::min(remainder, (std::size_t)INT_MAX);
      DRLOG("{}:{} win_.put {} bytes at off {}, dst offset {}",
            default_comm().rank(), __LINE__, s, off, offset + off);
      win_.put((uint8_t *)src + off, s, peer, offset + off);
      off += s;
      remainder -= s;
    }
#endif
  }

  std::size_t getrank() { return win_.communicator().rank(); }

  void fence() { win_.fence(); }

private:
  std::size_t get_peer(const std::size_t segment_index) { 
    const auto size = win_.communicator().size();
    return segment_index < size ? segment_index : 2 * size - segment_index - 1;
  }
};

/// distributed vector
template <typename T, class BackendT = DualMpiBackend> 
class dual_distributed_vector {

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using backend_type = BackendT;

  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename dual_distributed_vector::value_type;
    using difference_type = typename dual_distributed_vector::difference_type;

    iterator() {}
    iterator(const dual_distributed_vector *parent, difference_type offset)
        : parent_(parent), offset_(offset) {}

    auto operator+(difference_type n) const {
      return iterator(parent_, offset_ + n);
    }
    friend auto operator+(difference_type n, const iterator &other) {
      return other + n;
    }
    auto operator-(difference_type n) const {
      return iterator(parent_, offset_ - n);
    }
    auto operator-(iterator other) const { return offset_ - other.offset_; }

    auto &operator+=(difference_type n) {
      offset_ += n;
      return *this;
    }
    auto &operator-=(difference_type n) {
      offset_ -= n;
      return *this;
    }
    auto &operator++() {
      offset_++;
      return *this;
    }
    auto operator++(int) {
      auto old = *this;
      offset_++;
      return old;
    }
    auto &operator--() {
      offset_--;
      return *this;
    }
    auto operator--(int) {
      auto old = *this;
      offset_--;
      return old;
    }

    bool operator==(iterator other) const {
      if (parent_ == nullptr || other.parent_ == nullptr) {
        return false;
      } else {
        return offset_ == other.offset_;
      }
    }
    auto operator<=>(iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ <=> other.offset_;
    }

    auto operator*() const {
      auto segment_size = parent_->segment_size_;
      return parent_
          ->segments()[offset_ / segment_size][offset_ % segment_size];
    }
    auto operator[](difference_type n) const { return *(*this + n); }

    auto local() {
      auto segment_size = parent_->segment_size_;
      return (parent_->segments()[offset_ / segment_size].begin() +
              offset_ % segment_size)
          .local();
    }

    //
    // Support for distributed ranges
    //
    // distributed iterator provides segments
    // remote iterator provides local
    //
    auto segments() {
      return dr::__detail::drop_segments(parent_->segments(), offset_);
    }

  private:
    const dual_distributed_vector *parent_ = nullptr;
    difference_type offset_;
  };

  // Do not copy
  // We need a move constructor for the implementation of reduce algorithm
  dual_distributed_vector(const dual_distributed_vector &) = delete;
  dual_distributed_vector &operator=(const dual_distributed_vector &) = delete;
  dual_distributed_vector(dual_distributed_vector &&) { assert(false); }

  /// Constructor
  dual_distributed_vector(std::size_t size = 0, 
                          distribution dist = distribution()) {
    init(size, dist);
  }

  /// Constructor
  dual_distributed_vector(std::size_t size, value_type fill_value,
                          distribution dist = distribution()) {
    init(size, dist);
    mp::fill(*this, fill_value);
  }

  ~dual_distributed_vector() {
    if (finalized()) return;

    for (size_t i = 0; i < DUAL_SEGMENTS_PER_PROC; i++) {
      fence(i);

      if (datas_[i] != nullptr) {
        backends_[i].deallocate(datas_[i], data_size_ * sizeof(value_type));
      }

      delete halos_[i];
    }

    delete halo_;
  }

  /// Returns iterator to beginning
  auto begin() const { return iterator(this, 0); }
  /// Returns iterator to end
  auto end() const { return begin() + size_; }

  /// Returns size
  auto size() const { return size_; }
  /// Returns reference using index
  auto operator[](difference_type n) const { return *(begin() + n); }

  auto &halo() const { return *halo_; }

  auto segments() const { return rng::views::all(segments_); }
  // auto segments() { return rng::views::all(segments_); }

  __attribute__((unused))
  void fence(const std::size_t i) { backends_[i].fence(); }

  auto res_idx(const std::size_t segment_index) const {
    return segment_index < default_comm().size() ? 0 : 1;
  }

  backend_type& backend(const std::size_t segment_index) {
    return backends_[res_idx(segment_index)];
  }
  const backend_type& backend(const std::size_t segment_index) const {
    return backends_[res_idx(segment_index)];
  }

  T *data(const std::size_t segment_index) {
    return datas_[res_idx(segment_index)];
  }

private:
  void init(auto size, auto dist) {
    size_ = size;
    distribution_ = dist;

    // determine the distribution of data
    auto comm_size = default_comm().size(); // dr-style ignore
    auto hb = dist.halo();
    std::size_t gran = dist.granularity();
    // TODO: make this an error that is reported back to user
    assert(size % gran == 0 && "size must be a multiple of the granularity");
    assert(hb.prev % gran == 0 && "size must be a multiple of the granularity");
    assert(hb.next % gran == 0 && "size must be a multiple of the granularity");

    std::size_t segment_count = comm_size * DUAL_SEGMENTS_PER_PROC;
    auto proc_segments_size = gran * std::max({
        (size / gran + segment_count - 1) / segment_count,
        hb.prev / gran, 
        hb.next / gran});
    segment_size_ = proc_segments_size;

    std::size_t actual_segment_count_ = 
      size_ / segment_size_ + (size_ % segment_size_ == 0 ? 0 : 1);
    assert(actual_segment_count_ <= segment_count
      && "there must be at most 2 segments per process");

    data_size_ = segment_size_ + hb.prev + hb.next;

    std::cout << "creating dual_distributed vector\n"
      << "\tsize: " << size << "\n"
      << "\tsegment_size_: " << segment_size_ << "\n"
      << "\tactual_segment_count_: " << actual_segment_count_ << "\n"
      << "\tdata_size_: " << data_size_ << "\n";

    for (std::size_t i = 0; i < DUAL_SEGMENTS_PER_PROC; i++) {
      if (size_ > 0) {
        datas_.push_back(static_cast<T *>(
          backends_[i].allocate(data_size_ * sizeof(value_type))));
      }

      halos_.push_back(new span_halo<T>(default_comm(), datas_[i], data_size_, hb));
    }

    halo_ = new cyclic_span_halo<T>(halos_);

    std::size_t segment_index = 0;
    for (std::size_t i = 0; i < size; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, size - i), data_size_);
    }

    for (size_t i = 0; i < default_comm().size(); i++) {
      segments_[i].swap_state();
    }

    for (size_t i = 0; i < DUAL_SEGMENTS_PER_PROC; i++) {
      fence(i);
    }
  }

  friend dual_dv_segment_iterator<dual_distributed_vector>;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0; // size + halo

  std::vector<span_halo<T> *> halos_;
  std::vector<T *> datas_;
  cyclic_span_halo<T> *halo_;

  distribution distribution_;
  std::size_t size_;
  std::vector<dual_dv_segment<dual_distributed_vector>> segments_;
  std::vector<DualMpiBackend> backends_{DUAL_SEGMENTS_PER_PROC};
};

template <typename T, typename B>
auto &halo(const dual_distributed_vector<T, B> &dv) {
  return dv.halo();
}

} // namespace dr::mp
