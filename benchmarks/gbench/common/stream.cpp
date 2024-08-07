// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "stream.hpp"

using VecT = xp::distributed_vector<float>;
DR_BENCHMARK(Stream_Copy<VecT>)->Name("Stream_Copy_DR");
DR_BENCHMARK(Stream_Scale<VecT>)->Name("Stream_Scale_DR");
DR_BENCHMARK(Stream_Add<VecT>)->Name("Stream_Add_DR");
DR_BENCHMARK(Stream_Triad<VecT>)->Name("Stream_Triad_DR");
