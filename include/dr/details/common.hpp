namespace lib {

// Equal size partitions, exactly covers num
inline size_t partition(size_t num, size_t denom) {
  assert(num % denom == 0);
  return num / denom;
}

// Equal size partitions, rounding up to cover num
inline size_t partition_up(size_t num, size_t multiple) {
  return (num + multiple - 1) / multiple;
}

} // namespace lib
