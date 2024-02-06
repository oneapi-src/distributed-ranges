// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#ifdef DRISHMEM
#include "../common/stream.hpp"

using VectT = dr::mhp::distributed_vector<float, dr::mhp::IshmemBackend>;
DR_BENCHMARK(Stream_Copy<VectT>)->Name("Stream_Copy_ishmem");
DR_BENCHMARK(Stream_Scale<VectT>)->Name("Stream_Scale_ishmem");
DR_BENCHMARK(Stream_Add<VectT>)->Name("Stream_Add_ishmem");
DR_BENCHMARK(Stream_Triad<VectT>)->Name("Stream_Triad_ishmem");
#endif
