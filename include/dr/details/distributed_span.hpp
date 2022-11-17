namespace lib {

template <typename T, remote_contiguous_iterator Iter = remote_pointer<T>>
class distributed_span {
public:
  using local_span_type = remote_span<T, std::dynamic_extent, Iter>;

  using element_type = T;

  /// Type of values stored in the array
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  /// Type of pointers
  using pointer = Iter;

  /// Type of references
  using reference = std::iter_reference_t<Iter>;

  using joined_view_type =
      rng::join_view<rng::ref_view<std::vector<local_span_type>>>;

#ifdef DR_SPEC

  /// Type of distributed_span iterators
  using iterator = remote_contiguous_iterator;

#else

  /// Type of distributed_span iterators
  class iterator {};

#endif

  /// Create an empty distributed_span
  constexpr distributed_span() noexcept = default;

  constexpr distributed_span(const distributed_span &) noexcept = default;
  constexpr distributed_span &
  operator=(const distributed_span &) noexcept = default;

  /// Create a distributed span out of a range of remote_span objects
  template <rng::input_range R>
    requires(std::is_same_v<rng::range_value_t<R>, local_span_type>)
  constexpr distributed_span(R &&spans);

  /// Number of elements
  constexpr size_type size() const noexcept;

  /// Size of the span in bytes
  constexpr size_type size_bytes() const noexcept;

  /// Access individual elements
  constexpr reference operator[](size_type idx) const;

  constexpr std::size_t local_size(size_type rank = 0) const noexcept {
    return spans_[rank].size();
  }

  /// Whether the span is empty
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

  /// Retrieve a view of the subspans that comprise the distributed_span
  const std::span<remote_span<T>> &get_subspans() const noexcept;

  /// distributed_span representing elements [Offset, Offset + Count)
  constexpr distributed_span<element_type>
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const {
    std::vector<local_span_type> new_spans;

    size_type local_id = Offset / local_size();
    size_type local_offset = Offset % local_size();
    size_type local_count = std::min(local_size() - local_offset, Count);

    new_spans.push_back(spans_[local_id].subspan(local_offset, local_count));

    local_id++;

    for (; local_id * local_size() < Offset + Count && local_id < spans_.size();
         local_id++) {
      size_type local_count =
          std::min(local_size(), (Offset + Count) - local_id * local_size());
      new_spans.push_back(spans_[local_id].subspan(0, local_count));
    }

    return distributed_span<element_type>(new_spans);
  }

  /// distributed_span to first `Count` elements
  constexpr distributed_span<element_type> first(size_type Count) const {
    return subspan(0, Count);
  }

  /// distributed_span to last `Count` elements
  constexpr distributed_span<element_type> last(size_type Count) const {
    return subspan(size() - Count, Count);
  }

  /// Iterator to beginning
  iterator begin() const { return iterator(spans_, 0); }

  /// Iterator to end
  iterator end() const { return iterator(spans_, size()); }

  /*
    iterator local_begin() const {
      return local_joined_view_.begin();
    }
    iterator local_end() const {
      return local_joined_view_.end();
    }
    */

  /// First element
  constexpr reference front() const { return spans_.front().front(); }

  /// Last element
  constexpr reference back() const { return spans_.back().back(); }

private:
  std::vector<local_span_type> &
  set_local_views(const std::vector<local_span_type> &spans) {
    my_spans_.resize(0);
    for (auto &&span : spans_) {
      assert(false);
      if (span.data().rank == 0 // BCL::rank()
      ) {
        my_spans_.push_back(span);
      }
    }
    return my_spans_;
  }

private:
  std::size_t size_ = 0;
  std::vector<local_span_type> spans_;
  joined_view_type joined_view_;
  std::vector<local_span_type> my_spans_;
  joined_view_type local_joined_view_;
};

} // namespace lib
