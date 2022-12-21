// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

/// parallel ???
class parallel_explicit {};

class collective_root_policy {
public:
  collective_root_policy(int root) : root_(root) {}

  int root() { return root_; }

private:
  int root_;
};

} // namespace lib
