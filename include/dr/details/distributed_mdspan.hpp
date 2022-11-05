namespace lib {

template <typename DistributedVector> class distributed_accessor {
public:
  using element_type = typename DistributedVector::element_type;
  using offset_policy = distributed_accessor;
  using reference = typename DistributedVector::reference;
  using data_handle_type = typename DistributedVector::iterator;

  constexpr distributed_accessor() noexcept = default;
  constexpr reference access(data_handle_type p, size_t i) const {
    return p[i];
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }
};

template <typename T, typename Extents, typename Layout = stdex::layout_right,
          typename D = block_cyclic>
class distributed_mdspan {
private:
  using dspan = stdex::mdspan<T, Extents, Layout,
                              distributed_accessor<distributed_vector<T>>>;

public:
#ifdef DR_SPEC
  using reference = implementation_defined;
#else
  using reference = typename dspan::reference;
#endif
  using value_type = typename distributed_vector<T>::value_type;
  using size_type = typename distributed_vector<T>::size_type;
  using difference_type = typename distributed_vector<T>::difference_type;
  using iterator = typename distributed_vector<T>::iterator;

  /// Construct from a distributed_vector with the requested dimesions
  template <typename... Args>
  distributed_mdspan(distributed_vector<T> &dvector, Args... args)
      : dvector_(dvector),
        mdspan_(dvector.begin(), std::forward<Args>(args)...) {}

  /// first element in layout order
  auto begin() { return dvector_.begin(); }

  /// last element in layout order
  auto end() { return dvector_.end(); }

  /// fence for updates
  void fence() { return dvector_.fence(); }

  /// segments of mdspan
  auto segments() { return dvector_.segments(); }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) {
    return mdspan_(std::forward<Args>(args)...);
  }

private:
  distributed_vector<T> &dvector_;
  dspan mdspan_;
};

template <typename T, typename Extents, typename Layout = stdex::layout_right,
          typename D = block_cyclic>
class distributed_mdarray : public distributed_vector<T> {
private:
  using dspan = stdex::mdspan<T, Extents, Layout,
                              distributed_accessor<distributed_vector<T>>>;

public:
#ifdef DR_SPEC
  /// Reference to element
  using reference = implementation_defined;
#else
  using reference = typename dspan::reference;
#endif

  /// Construct an mdarray with requested dimensions
  template <typename... Args>
  distributed_mdarray(Args... args)
      : distributed_vector<T>(product(std::forward<Args>(args)...)),
        mdspan_(this->begin(), std::forward<Args>(args)...) {}

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) {
    return mdspan_(std::forward<Args>(args)...);
  }

private:
  dspan mdspan_;
};

} // namespace lib
