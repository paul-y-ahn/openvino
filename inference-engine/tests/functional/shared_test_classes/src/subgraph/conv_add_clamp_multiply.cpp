// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv_add_clamp_multiply.hpp"

namespace SubgraphTestsDefinitions {

std::string ConvAddClampMultiply::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ConvAddClampMultiply::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto weights = CommonTestUtils::generate_float_numbers(128 * inputShape[1], -0.2f, 0.2f);
    auto conv = ngraph::builder::makeConvolution(params[0], ngPrc, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                 ngraph::op::PadType::VALID, 128, false, weights);
    std::vector<size_t> input_dims = { 1, 1, 1, 1 };
    std::vector<float> clamp_min_max = { -20, 20 };
    auto shift = CommonTestUtils::generate_float_numbers(input_dims[0], 0.0f, 20.0f);
    auto add_const = ngraph::builder::makeConstant(ngPrc, input_dims, shift);
    auto add = ngraph::builder::makeEltwise(conv, add_const, ngraph::helpers::EltwiseTypes::ADD);
    auto clamp = std::make_shared<ngraph::opset1::Clamp>(add, clamp_min_max[0], clamp_min_max[1]);
    auto mul = ngraph::builder::makeEltwise(conv, clamp, ngraph::helpers::EltwiseTypes::MULTIPLY);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mul)};
    function = std::make_shared<ngraph::Function>(results, params, "ConvAddClampMultiply");
}

}  // namespace SubgraphTestsDefinitions