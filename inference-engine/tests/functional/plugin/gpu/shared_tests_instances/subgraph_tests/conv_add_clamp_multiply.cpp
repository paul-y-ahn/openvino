// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/conv_add_clamp_multiply.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_CASE_P(smoke_NoReshape, ConvAddClampMultiply,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({1, 128, 40, 40})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvAddClampMultiply::getTestCaseName);

}  // namespace