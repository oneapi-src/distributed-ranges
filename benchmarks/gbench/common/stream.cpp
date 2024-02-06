// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "stream.hpp"

using VecT = xhp::distributed_vector<float>;
DR_BENCHMARK(Stream_Copy<VecT>)->Name("Stream_Copy");
DR_BENCHMARK(Stream_Scale<VecT>)->Name("Stream_Scale");
DR_BENCHMARK(Stream_Add<VecT>)->Name("Stream_Add");
DR_BENCHMARK(Stream_Triad<VecT>)->Name("Stream_Triad");
