// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

inline std::size_t partition_up(std::size_t num, std::size_t multiple) {
  return (num + multiple - 1) / multiple;
}

template <typename Seq>
void set_step(Seq &seq, typename Seq::value_type v = 0,
              typename Seq::value_type step = 1) {
  for (auto &r : seq) {
    r = v;
    v += step;
  }
}

int check(const auto &actual, const auto &reference, int max_errors = 10) {
  int errors = 0;

  for (std::size_t i = 0; i < std::min(actual.size(), reference.size()); i++) {
    if (actual[i] != reference[i]) {
      if (errors == 0)
        fmt::print("Value mismatches (actual):(reference)\n");
      if (errors < max_errors)
        fmt::print("  {}: {}:{}\n", i, actual[i], reference[i]);
      errors++;
    }
  }
  if (actual.size() != reference.size()) {
    fmt::print("Size mismatch: {}(actual) {}(reference)\n", actual.size(),
               reference.size());
    errors++;
  }

  return errors;
}
