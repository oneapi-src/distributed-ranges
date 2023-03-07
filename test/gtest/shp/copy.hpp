#pragma once

#include "shp-tests.hpp"

template <typename AllocT> class CopyTest : public testing::Test {
public:
  using DistVec = shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
};

using AllocatorTypes =
    ::testing::Types<shp::device_allocator<int>,
                     shp::shared_allocator<long long unsigned int>>;
