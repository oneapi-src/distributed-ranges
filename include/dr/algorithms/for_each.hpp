// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace lib {

/// Iterate over a distributed range
template <class ExecutionPolicy, typename R, class UnaryFunction>
void for_each(ExecutionPolicy &&policy, R range, UnaryFunction f) {
  assert(false);
}

} // namespace lib
