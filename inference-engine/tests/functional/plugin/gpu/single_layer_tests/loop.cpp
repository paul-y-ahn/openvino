// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ngraph/op/util/attr_types.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/single_layer/loop.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

using LoopParamsGPU = typename std::tuple<
        bool,                                                              // ExecuteFirstIteration
        bool,                                                              // BodyCondition is a constant?
        bool,                                                              // BodyCondition value, if it is a Const
        int64_t,                                                           // TripCount, -1 means infinity
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>,         // inputs
        InferenceEngine::Precision>;                                       // Network precision

class LoopTestGPU : public testing::WithParamInterface<LoopParamsGPU>,
                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParamsGPU> &obj);

protected:
    void SetUp() override;
};


using StaticShapeLoopParamsGPU = typename std::tuple<
        bool,
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        InferenceEngine::SizeVector,
        InferenceEngine::Precision>;

/**
 * Test case with static SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class StaticShapeLoopTestGPU : public testing::WithParamInterface<StaticShapeLoopParamsGPU>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeLoopParamsGPU> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> PredefinedRefs();

private:
    bool unrolling;             // unroll Loop
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    InferenceEngine::SizeVector data_shape;
    InferenceEngine::Precision data_prc;

    int64_t actual_n_iter();

protected:
    void SetUp() override;
};


class TrivialLoopTestGPU : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    using RefBlobGenerator = std::function<InferenceEngine::Blob::Ptr (const InferenceEngine::TensorDesc &info)>;
    std::map<std::string, RefBlobGenerator> inputGens, outputGens;

    void CreateSlicedLoop(size_t batch_size, size_t num_iteration, InferenceEngine::Precision iePrc,
                          InferenceEngine::SizeVector& ieShape);
    void CreateSlicedLoopDynCondition(size_t batch_size, size_t num_iteration, InferenceEngine::Precision iePrc,
                          InferenceEngine::SizeVector& ieShape, size_t trip_count);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        auto found = inputGens.find(info.name());
        if (found != inputGens.end()) {
            return found->second(info.getTensorDesc());
        }

        found = inputGens.find("");
        if (found != inputGens.end()) {
            return found->second(info.getTensorDesc());
        }

        return LayerTestsCommon::GenerateInput(info);
    }

    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override {
        if (outputGens.empty())
            return LayerTestsCommon::CalculateRefs();

        const auto results = function->get_results();
        const auto outs_info = cnnNetwork.getOutputsInfo();
        const auto num_out_blob = results.size();

        std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> res_collection(num_out_blob);

        for (int i = 0; i < num_out_blob; i++) {
            // TODO: name of original NG result doesn't match with outs after conversion.
            //       Expected : auto name = results[i]->get_friendly_name();
            auto name = results[i]->get_input_node_ptr(0)->get_friendly_name();
            auto data = outs_info.at(name);
            IE_ASSERT(data != nullptr);

            RefBlobGenerator generator;
            auto found = outputGens.find(name);
            if (found != outputGens.end()) {
                generator = found->second;
            } else {
                found = outputGens.find("");
                if (found != outputGens.end()) {
                    generator = found->second;
                }
            }

            IE_ASSERT(generator != nullptr) << "Test output generator is not specified";
            auto blob = generator(data->getTensorDesc());
            auto blob_size = blob->byteSize();
            auto blob_ptr = blob->buffer().as<uint8_t*>();

            auto &res = res_collection[i];
            res.second.resize(blob_size);
            std::copy(blob_ptr, blob_ptr + blob_size, res.second.begin());
        }
        return res_collection;
    }
};

    std::string LoopTestGPU::getTestCaseName(const testing::TestParamInfo<LoopParamsGPU> &obj) {
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition; // works only if is_body_condition_const ==
        int64_t trip_count;
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>> inputs;
        InferenceEngine::Precision netPrecision;
        std::tie(execute_first_iteration, is_body_condition_const, body_condition,
            trip_count, inputs, netPrecision) = obj.param;

        std::vector<std::vector<size_t>> inputs_separate;
        std::vector<LOOP_IN_TYPE> types_separate;
        for (auto &el : inputs) {
            inputs_separate.push_back(el.first);
            types_separate.push_back(el.second);
        }
        std::ostringstream result;
        result << "execute_first_iteration" << execute_first_iteration << "_";
        result << "is_body_condition_const=" << is_body_condition_const << "_";
        result << "body_condition=" << body_condition << "_";
        result << "trip_count=" << trip_count << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputs_separate) << "_";
        result << "types=" << CommonTestUtils::vec2str(types_separate) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

    void LoopTestGPU::SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition; // works only if is_body_condition_const ==
        int64_t trip_count;
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>> inputs;
        InferenceEngine::Precision netPrecision;
        targetDevice = "GPU";
        std::tie(execute_first_iteration, is_body_condition_const, body_condition,
            trip_count, inputs, netPrecision) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        // That which we iterate over
        std::vector<std::vector<size_t>> inputs_separate;
        std::vector<LOOP_IN_TYPE> types_separate;
        for (auto &el : inputs) {
            inputs_separate.push_back(el.first);
            types_separate.push_back(el.second);
        }
        // Example:
        /*      auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{32, 1, 10});
        auto Y = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{32, 1, 10});
        auto M = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{32, 1, 10});*/
        auto params = ngraph::builder::makeParams(ngPrc, inputs_separate);

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(inputs_separate.size(), ngraph::PartialShape::dynamic());
        auto current_iteration = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{1});

        //Example:
/*      auto Xi = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto Yi = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto M_body = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());*/

        ngraph::ParameterVector body_params;
        for (const auto &pshape : body_params_shapes) {
            auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, pshape);
            body_params.push_back(paramNode);
        }

        std::shared_ptr<ngraph::Node> body_condition_const;
        if (is_body_condition_const) {
            if (body_condition) {
                body_condition_const = std::make_shared<ngraph::opset5::Constant>(
                        ngraph::element::boolean, ngraph::Shape{1}, true);
            } else {
                body_condition_const = std::make_shared<ngraph::opset5::Constant>(
                        ngraph::element::boolean, ngraph::Shape{1}, false);
            }
        }

        auto trip_count_const =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, trip_count);

        std::shared_ptr<ngraph::Node> exec_condition;
        if (execute_first_iteration) {
            exec_condition = std::make_shared<ngraph::opset5::Constant>(
                    ngraph::element::boolean, ngraph::Shape{1}, true);
        } else {
            exec_condition = std::make_shared<ngraph::opset5::Constant>(
                    ngraph::element::boolean, ngraph::Shape{1}, false);
        }

        // Body
        std::shared_ptr<ngraph::Node> Zo = body_params[0];
        for (int i = 1; i < body_params.size(); ++i) {
            Zo = std::make_shared<ngraph::op::v1::Add>(body_params[i], Zo);
        }

        // body_params.insert(body_params.begin(), current_iteration);
        auto body = std::make_shared<ngraph::Function>(ngraph::OutputVector{body_condition_const, Zo},
                                                  body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(trip_count_const, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        for (int i = 0; i < body_params.size(); ++i) {
            if (types_separate[i] == LOOP_IN_TYPE::INVARIANT) {
                loop->set_invariant_input(body_params[i], params[i]);
            } else if (types_separate[i] == LOOP_IN_TYPE::MERGED) {
                // todo: support several merged inputs
                // now supported only one in this sample
                loop->set_merged_input(body_params[i], params[i], Zo);
            }
        }

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

        auto result0 = std::make_shared<ngraph::opset5::Result>(out0);
        auto result1 = std::make_shared<ngraph::opset5::Result>(out1);
        auto result2 = std::make_shared<ngraph::opset5::Result>(out2);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result0, result1, result2}, params, "loop");
    }

    void StaticShapeLoopTestGPU::SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        auto args_papck = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
        std::tie(
            unrolling,
            static_continue_cond,
            args_papck,
            start_value,
            data_shape,
            data_prc) = GetParam();
        targetDevice = "GPU";

        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_prc);
        const auto ngShape = ngraph::Shape{data_shape};
        const auto scalarShape = ngraph::Shape{};

        ngraph::ParameterVector params{};
        auto cond_input_create = [&params] (ngraph::element::Type prc, const ngraph::Shape &shape, int value = 0, bool is_static = false)
                -> std::shared_ptr<ngraph::Node> {
            if (is_static)
                return std::make_shared<ngraph::opset5::Constant>(prc, shape, value);

            auto input = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
            params.push_back(input);
            return input;
        };

        auto start = cond_input_create(prc, ngShape);
        auto count = cond_input_create(ngraph::element::i64, scalarShape, max_iter_num, static_iter_num);
        auto skip  = cond_input_create(ngraph::element::boolean, scalarShape, true, static_continue_cond);

        //
        //      count skip  start         count skip      start
        //                  /                             /
        //          ___*___*____           __________*___*____       | idx  | data | out |
        //         |  idx  in   |         | ex_val  idx  in   |      |  0   |  7   |  7  |
        //         |   |  /     |         |   |   /  |  /     |      |  1   |  7   |  8  |
        //         |   add      |         |   less   add      |      |  2   |  8   |  10 |
        //         |   |   true |         |    |     |        |      |  3   |  10  |  13 |
        //         |   |    |   |         |    |     |        |       ~~~~~  * * *  ~~~~~
        //         |  out  cnd  |         |   cnd   out       |
        //         |___*____*___|         |____*_____*________|
        //           Full loop              Dynamic exit loop
        //           n_iter = count         n_iter = ex_val
        //
        auto b_indx = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{});
        auto b_data = std::make_shared<ngraph::opset5::Parameter>(prc, ngShape);
        auto b_indx_cast = std::make_shared<ngraph::opset5::Convert>(b_indx, prc);
        auto b_add  = std::make_shared<ngraph::opset5::Add>(b_data, b_indx_cast);

        std::shared_ptr<ngraph::Node> b_cond;
        if (dynamic_exit == -1) {
            b_cond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
        } else {
            auto b_exit_value = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, dynamic_exit);
            b_cond = std::make_shared<ngraph::opset5::Less>(b_indx, b_exit_value);
        }
#if 1
        auto body = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {b_cond, b_add},    // TODO: check with reverse
                ngraph::ParameterVector {b_indx, b_data});  // TODO: check with reverse
#else
        auto b_cond_output = std::make_shared<ngraph::opset1::Result>(b_cond, true);
        auto b_add_output = std::make_shared<ngraph::opset1::Result>(b_add, true);

        auto body = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {b_cond_output, b_add_output},    // TODO: check with reverse
                ngraph::ParameterVector {b_indx, b_data});  // TODO: check with reverse
#endif
        {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::VisualizeTree>("body.svg");
            manager.run_passes(body);
        }

        auto loop = std::make_shared<ngraph::opset5::Loop>(count, skip);
        loop->set_function(body);
        loop->set_special_body_ports({0, 0});
        loop->set_merged_input(b_data, start, b_add);
        if (axis == -1)
            loop->get_iter_value(b_add, -1);
        else
            loop->get_concatenated_slices(b_add, 0, 1, 1, -1, axis);

        function = std::make_shared<ngraph::Function>(
                ngraph::OutputVector {loop},
                params);
        if (unrolling) {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::UnrollTensorIterator>();
            manager.run_passes(function);
        }
        {
            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::VisualizeTree>("main.svg");
            manager.run_passes(function);
        }
    }

    InferenceEngine::Blob::Ptr StaticShapeLoopTestGPU::GenerateInput(const InferenceEngine::InputInfo &info) const {
        auto tdesc = info.getTensorDesc();
        auto blob = make_blob_with_precision(tdesc);
        blob->allocate();

        if (tdesc.getLayout() == InferenceEngine::SCALAR) {
            auto scalar_1d = CommonTestUtils::make_reshape_view(blob, {1});
            CommonTestUtils::fill_data_with_broadcast(scalar_1d, 0, {static_cast<float>(max_iter_num)});
        } else {
            CommonTestUtils::fill_data_with_broadcast(blob, 0, {static_cast<float>(start_value)});
        }

        return blob;
    }

    int64_t StaticShapeLoopTestGPU::actual_n_iter() {
        constexpr auto INF_N_ITER = std::numeric_limits<int64_t>::max();
        IE_ASSERT(dynamic_exit != -1 || max_iter_num != -1);

        // dynamic_exit + 1 - because loop body looks like do-while loop with post condition check.
        return std::min(dynamic_exit == -1 ? INF_N_ITER : dynamic_exit + 1,
                        max_iter_num == -1 ? INF_N_ITER : max_iter_num);
    }

    // Predefined ref output
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> StaticShapeLoopTestGPU::PredefinedRefs() {
        bool auto_concat_out = (axis != -1);
        const auto n_iter = actual_n_iter();

        auto ref_shape = data_shape;
        if (auto_concat_out)
            ref_shape[axis] *= n_iter;

        using namespace CommonTestUtils;
        InferenceEngine::TensorDesc tdesc {data_prc, ref_shape, InferenceEngine::TensorDesc::getLayoutByDims(ref_shape)};
        std::pair<ngraph::element::Type, std::vector<uint8_t>> res;
        res.first = function->get_result()->get_element_type();
        res.second = std::vector<uint8_t>(byte_size(tdesc));
        auto out = make_blob_with_precision(tdesc, res.second.data());

        std::vector<float> vals(n_iter);
        float val = start_value;
        for (int i = 0; i < n_iter; i++) {
            val += i;
            vals[i] = val;
        }

        if (auto_concat_out)
            fill_data_with_broadcast(out, axis, vals);
        else
            fill_data_with_broadcast(out, 0, {val});  // broadcast scalar data

        return {res};
    }

    void TrivialLoopTestGPU::CreateSlicedLoop(size_t batch_size, size_t num_iteration, InferenceEngine::Precision iePrc,
                                           InferenceEngine::SizeVector& ieShape) {
        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iePrc);
        const auto scalarShape = ngraph::Shape{};

        auto shape = ngraph::Shape{ieShape};
        auto to_slice_shape = ngraph::Shape{ieShape};
        to_slice_shape[0] = batch_size;

        auto to_slice = std::make_shared<ngraph::opset5::Parameter>(prc, to_slice_shape);
        auto start = std::make_shared<ngraph::opset5::Constant>(prc, shape, 0);
        auto count = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, num_iteration);
        auto icond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, scalarShape, true);

        // Loop body
        auto b_data = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
        auto b_recu = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
        auto b_add  = std::make_shared<ngraph::opset5::Add>(b_data, b_recu);
        auto b_cond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, scalarShape, true);

        auto body = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {b_cond, b_add},
                ngraph::ParameterVector {b_data, b_recu});

        auto loop = std::make_shared<ngraph::opset5::Loop>(count, icond);
        loop->set_function(body);
        loop->set_special_body_ports({-1, 0});
        loop->set_sliced_input(b_data, to_slice, 0, 1, 1, -1, 0);
        loop->set_merged_input(b_recu, start, b_add);
        loop->get_iter_value(b_add, -1);

        function = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {loop},
                ngraph::ParameterVector {to_slice});
    }

    void TrivialLoopTestGPU::CreateSlicedLoopDynCondition(size_t batch_size, size_t num_iteration, InferenceEngine::Precision iePrc,
                                           InferenceEngine::SizeVector& ieShape, size_t trip_count) {
        auto shape = ngraph::Shape{ieShape};
        auto to_slice_shape = ngraph::Shape{ieShape};
        to_slice_shape[0] = batch_size;

        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iePrc);
        const auto scalarShape = ngraph::Shape{};

        auto to_slice = std::make_shared<ngraph::opset5::Parameter>(prc, to_slice_shape);
        auto start = std::make_shared<ngraph::opset5::Constant>(prc, shape, 0);
        auto exit_on = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, num_iteration);
        auto count = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, trip_count);
        auto icond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, scalarShape, true);

        // Loop body
        auto b_data = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
        auto b_recu = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
        auto b_add  = std::make_shared<ngraph::opset5::Add>(b_data, b_recu);
        auto b_iter = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, scalarShape);
        auto b_exit_on = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, scalarShape);
        auto b_cond = std::make_shared<ngraph::opset5::Less>(b_iter, b_exit_on);

        auto body = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {b_cond, b_add},
                ngraph::ParameterVector {b_data, b_recu, b_iter, b_exit_on});

        auto loop = std::make_shared<ngraph::opset5::Loop>(count, icond);
        loop->set_function(body);
        loop->set_special_body_ports({2, 0});
        loop->set_sliced_input(b_data, to_slice, 0, 1, 1, -1, 0);
        loop->set_invariant_input(b_exit_on, exit_on);
        loop->set_merged_input(b_recu, start, b_add);
        loop->get_iter_value(b_add, -1);

        function = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {loop},
                ngraph::ParameterVector {to_slice});
    }

    TEST_P(LoopTestGPU, CompareWithRefs) {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        Run();
    }

    TEST_P(StaticShapeLoopTestGPU, CompareWithRefs) {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        Run();
    }
}  // namespace LayerTestsDefinitions

namespace {
    // // without clip values increase rapidly, so use only seq_lengths = 2
    // std::vector<bool> execute_first_iteration{true};
    // std::vector<bool> is_body_condition_const{true/*, false*/};
    // std::vector<bool> body_condition{true/*, false*/}; // works only if is_body_condition_const == true
    // std::vector<int64_t> trip_count{1, 10/*, -1*/}; // -1 means infinity
    // std::vector<std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>> inputs = {
    //         {{{4, 1, 10}, LOOP_IN_TYPE::INVARIANT}, {{4, 1, 10}, LOOP_IN_TYPE::MERGED}, {{4, 1, 10}, LOOP_IN_TYPE::MERGED}},
    // };
    // std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
    //                                                          InferenceEngine::Precision::FP16};

    // INSTANTIATE_TEST_CASE_P(smoke_LoopCommonZeroClip, LoopTestGPU,
    //                         ::testing::Combine(
    //                                 ::testing::ValuesIn(execute_first_iteration),
    //                                 ::testing::ValuesIn(is_body_condition_const),
    //                                 ::testing::ValuesIn(body_condition),
    //                                 ::testing::ValuesIn(trip_count),
    //                                 ::testing::ValuesIn(inputs),
    //                                 ::testing::ValuesIn(netPrecisions)),
    //                         LoopTestGPU::getTestCaseName);

    static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> static_loop_types {
            //  GCC4.8 limitation: have to specify type of each element in list
            //                               static_trip_count |  max | dynamic_exit | axis
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1, -1 },  // n_iter 5, no dynamic exit
            std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 },  // n_iter 5, const for loop with auto concatenated out
    };

    INSTANTIATE_TEST_CASE_P(smoke_StaticShapeLoop, StaticShapeLoopTestGPU,
                            testing::Combine(
                            /* unrolling */ testing::ValuesIn(std::vector<bool>{false}),
                            /* static_continue_cond */ testing::Values(true),
                            /* args_papck */ testing::ValuesIn(static_loop_types),
                            /* start_value */ testing::Values<int64_t>(0),
                            /* data_shape */ testing::Values<InferenceEngine::SizeVector>({2, 1, 4}),
                            /* data_prc */ testing::Values<InferenceEngine::Precision>(Precision::FP32, Precision::I32)));
//     INSTANTIATE_TEST_CASE_P(smoke_TrivialLoop, TrivialLoopTestGPU,
//                             testing::Combine(
//                                     testing::Values<InferenceEngine::Precision>(Precision::FP32, Precision::I32),
//                                     testing::Values<InferenceEngine::SizeVector>({2, 3, 4}),
//                                     testing::Values(CommonTestUtils::DEVICE_GPU)));

}  // namespace
