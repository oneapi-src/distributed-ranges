inline size_t partition_up(size_t num, size_t multiple) {
  return (num + multiple - 1) / multiple;
}

template <typename Seq> void show(std::string title, const Seq &seq) {
  std::cout << title;
  for (auto v : seq) {
    std::cout << " " << v;
  }
  std::cout << "\n";
}

template <typename Seq>
void set_step(Seq &seq, typename Seq::value_type v = 0,
              typename Seq::value_type step = 1) {
  for (auto &r : seq) {
    r = v;
    v += step;
  }
}

template <typename Seq>
int check(Seq &actual, Seq &reference, int max_errors = 10) {
  int errors = 0;

  for (size_t i = 0; i < std::min(actual.size(), reference.size()); i++) {
    if (actual[i] != reference[i]) {
      if (errors == 0)
        std::cout << "Value mismatches (actual):(reference)\n";
      if (errors < max_errors)
        std::cout << "  " << i << ": " << actual[i] << ":" << reference[i]
                  << "\n";
      errors++;
    }
  }
  if (actual.size() != reference.size()) {
    std::cout << "Size mismatch: " << actual.size() << "(actual) "
              << reference.size() << "(reference)\n";
    errors++;
  }

  return errors;
}
