namespace lib {

template <typename DistributedVector> class distributed_accessor {
public:
  using element_type = typename DistributedVector::element_type;
  using offset_policy = distributed_accessor;
  using reference = typename DistributedVector::reference;
  using data_handle_type = typename DistributedVector::iterator;

  constexpr distributed_accessor() noexcept = default;
  constexpr reference access(data_handle_type p, size_t i) const {
    drlog.debug(nostd::source_location::current(),
                "distributed_accessor const reference\n");
    return p[i];
  }

  reference access(data_handle_type p, size_t i) {
    drlog.debug(nostd::source_location::current(),
                "distributed_accessor reference\n");
    return p[i];
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }
};

// Assumes partition on leading dimension
template <typename Extents>
Extents local_extents(Extents extents, std::size_t comm_size) {
  std::array<typename Extents::index_type, extents.rank()> local;
  local[0] = partition_up(extents.extent(0), comm_size);
  for (std::size_t i = 1; i < Extents::rank(); i++) {
    local[i] = extents.extent(i);
  }

  return local;
}

template <typename T, typename Extents, typename Layout = stdex::layout_right>
class distributed_mdspan {
private:
  using dvector = distributed_vector<T>;
  using dmdspan =
      stdex::mdspan<T, Extents, Layout, distributed_accessor<dvector>>;

public:
#ifdef DR_SPEC
  /// Reference type
  using reference = implementation_defined;
  /// Value type
  using value_type = implementation_defined;
  /// Size type
  using size_type = implementation_defined;
  /// Difference type
  using difference_type = implementation_defined;
  /// Iterator
  using iterator = implementation_defined;
#else
  using reference = typename dmdspan::reference;
  using value_type = typename dvector::value_type;
  using size_type = typename dvector::size_type;
  using difference_type = typename dvector::difference_type;
  using iterator = typename dvector::iterator;
#endif
  /// Extents type
  using extents_type = Extents;
  /// Local segment type
  using local_type = stdex::mdspan<
      T, stdex::dextents<typename Extents::index_type, Extents::rank()>,
      Layout>;
  using accessor_type = typename dmdspan::accessor_type;
  using layout_type = typename dmdspan::layout_type;

  /// Construct from a distributed_vector with the requested dimensions
  template <typename... Args>
  distributed_mdspan(distributed_vector<T> &dvector, Args... args)
      : extents_(std::forward<Args>(args)...), dvector_(dvector),
        dmdspan_(dvector.begin(), extents_),
        local_mdspan_(dvector_.local().data(),
                      local_extents(extents_, comm_.size())) {
    assert(storage_size(extents_, comm_.size()) <= dvector.size());
  }

  /// Returns local segment
  local_type local() const { return local_mdspan_; }

  /// first element in layout order
  auto begin() { return dvector_.begin(); }

  /// last element in layout order
  auto end() { return dvector_.end(); }

  /// fence for updates
  void fence() { return dvector_.fence(); }

  /// segments of distributed_mdspan
  auto segments() { return dvector_.segments(); }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) {
    return dmdspan_(std::forward<Args>(args)...);
  }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) const {
    return dmdspan_(std::forward<Args>(args)...);
  }

private:
  communicator comm_;
  Extents extents_;
  dvector &dvector_;
  dmdspan dmdspan_;
  local_type local_mdspan_;
};

template <typename T, typename Extents, typename Layout = stdex::layout_right>
class distributed_mdarray {
private:
  using dvector = distributed_vector<T>;
  using dmdspan =
      stdex::mdspan<T, Extents, Layout, distributed_accessor<dvector>>;

public:
#ifdef DR_SPEC
  /// Reference to element
  using reference = implementation_defined;
  /// Reference to element
  using value_type = implementation_defined;
  /// Size type
  using size_type = implementation_defined;
  /// Difference type
  using difference_type = implementation_defined;
#else
  using reference = typename dmdspan::reference;
  using value_type = typename dvector::value_type;
  using size_type = typename dvector::size_type;
  using difference_type = typename dvector::difference_type;
  using iterator = typename dvector::iterator;
#endif
  /// Extents type
  using extents_type = Extents;
  /// Type of local segment
  using local_type = stdex::mdspan<
      T, stdex::dextents<typename Extents::index_type, Extents::rank()>,
      Layout>;
  using accessor_type = typename dmdspan::accessor_type;
  using layout_type = typename dmdspan::layout_type;

  /// Construct a distributed_mdarray with requested dimensions
  template <typename... Args>
  distributed_mdarray(Args... args)
      : extents_(std::forward<Args>(args)...),
        dvector_(storage_size(extents_, comm_.size())),
        dmdspan_(dvector_.begin(), extents_),
        local_mdspan_(dvector_.local().data(),
                      local_extents(extents_, comm_.size())) {}

  /// Returns local segment
  local_type local() const { return local_mdspan_; }

  /// first element in layout order
  auto begin() { return dvector_.begin(); }

  /// last element in layout order
  auto end() { return dvector_.end(); }

  /// fence for updates
  void fence() { return dvector_.fence(); }

  /// segments of distributed_mdspan
  auto segments() { return dvector_.segments(); }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) {
    drlog.debug(nostd::source_location::current(), "mdarray reference\n");
    return dmdspan_(std::forward<Args>(args)...);
  }

  /// multidimensional index operator
  template <typename... Args> reference operator()(Args... args) const {
    drlog.debug(nostd::source_location::current(), "mdarray reference const\n");
    return dmdspan_(std::forward<Args>(args)...);
  }

  /// Returns extents
  auto extents() const { return dmdspan_.extents(); }

private:
  communicator comm_;
  Extents extents_;
  dvector dvector_;
  dmdspan dmdspan_;
  local_type local_mdspan_;
};

} // namespace lib
