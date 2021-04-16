// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "cldnn_engine.h"

#include <cpp/ie_cnn_network.h>

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"

#include "api/loop.hpp"
#include "api/mutable_data.hpp"

#include <vector>
#include <algorithm>

using TensorIterator = ngraph::op::v0::TensorIterator;

namespace CLDNNPlugin {

static cldnn::mutable_data CreateIntData(Program &p, const cldnn::primitive_id& id, int32_t num) {
    auto mem = cldnn::memory::allocate(p.GetEngine(),
        { cldnn::data_types::i32, cldnn::format::bfyx, { 1, 1, 1, 1 } });
    auto ptr = mem.pointer<int32_t>();
    *ptr.begin() = num;
    return {id, mem};
}

static cldnn::primitive_id GetOutputPrimitiveID(const Program& p, const std::shared_ptr<ngraph::Node>& op) {
    cldnn::primitive_id output_id = layer_type_lower(op) + ":" + op->get_friendly_name();
    auto found = std::find_if(p.primitiveIDs.begin(), p.primitiveIDs.end(),
        [&output_id, &p](const std::pair<cldnn::primitive_id, cldnn::primitive_id>& pm) {
            return pm.second == output_id;
        });
    assert(found != p.primitiveIDs.end());
    return found->first;
}

static cldnn::primitive_id GetPrimitiveID(const Program& p, const std::shared_ptr<ngraph::Node>& op) {
    auto found = std::find_if(p.primitivesToIRLayersMap.begin(), p.primitivesToIRLayersMap.end(),
        [&op, &p](const std::pair<cldnn::primitive_id, std::vector<std::string>>& pm) {
            assert(pm.second.size() == 1);
            return pm.second.front() == op->get_friendly_name();
            // if (pm.second.front() == op->get_friendly_name()) {
            //     return true;
            // }
            // if (pm.second.front() == nameWithType) {
            //     return true;
            // }

            // cldnn::primitive_id newTarget = nameWithType;
            // while (p.prevPrimitiveIDs.count(newTarget)) {
            //     const auto newNames = p.prevPrimitiveIDs.at(newTarget);
            //     assert(newNames.size() == 1);
            //     if (pm.second.front() == newNames.front())
            //         return true;
            //     newTarget = newNames.front();
            // }
            // return false;
        });
    assert(found != p.primitivesToIRLayersMap.end());
    return found->first;
}

void CreateTensorIteratorOp(Program &p, const std::shared_ptr<TensorIterator> &op) {
    // loop can takes multiple inputs, no p.ValidateInputs()
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op); // equals to outputPrimitiveId

    // setup cldnn::data for outer network
    const cldnn::primitive_id trip_count_id = layerName + ":trip_count";
    const int32_t num_iterations = op->get_num_iterations();
    assert(num_iterations >= 0);
    {
        cldnn::mutable_data trip_count = CreateIntData(p, trip_count_id, num_iterations);
        p.AddPrimitive(trip_count);
    }
    const cldnn::primitive_id execution_condition_id = layerName + ":initial_execution_condition";
    {
        cldnn::mutable_data execution_condition = CreateIntData(p, execution_condition_id, 1);
        p.AddPrimitive(execution_condition);
    }
    const cldnn::primitive_id num_iteration_id = layerName + ":num_iteration";
    {
        cldnn::mutable_data num_iteration = CreateIntData(p, num_iteration_id, 0);
        p.AddPrimitive(num_iteration);
    }

    // set body topology
    InferenceEngine::CNNNetwork body_network(op->get_body());
    Program body_program(body_network, p.GetEnginePtr(), p.GetConfig(), false);
    auto body_topology = *body_program.GetTopology();

    // setup primitive_map and back_edges
    const auto& input_mappings = op->get_input_descriptions();
    const auto& output_mappings = op->get_output_descriptions();
    const auto& body_inputs = op->get_body()->get_parameters();
    const auto& body_outputs = op->get_body()->get_results();

    std::vector<cldnn::loop::primitive_mapping> primitive_map;
    std::vector<cldnn::loop::backedge_mapping> back_edges;

    for (const auto& input_mapping : input_mappings) {
        const cldnn::primitive_id& external_id = inputPrimitives.at(input_mapping->m_input_index);
        const auto& body_input = body_inputs.at(input_mapping->m_body_parameter_index);
        const cldnn::primitive_id internal_id = GetPrimitiveID(body_program, body_input);
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<TensorIterator::SliceInputDescription>(input_mapping)) {
            // input with iteration axis
            // primitive_mapping(primitive_type type, primitive_id external_id, primitive_id internal_id,
            // int32_t axis = -1, int32_t start = 0, int32_t end = -1, int32_t stride = 1)
            primitive_map.emplace_back(cldnn::loop::INPUT, external_id, internal_id,
                sliceInfo->m_axis, sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
        } else {
            // InvariantInputDescription or InputDescription
            // input without iteration axis
            primitive_map.emplace_back(cldnn::loop::INPUT, external_id, internal_id);
        }
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<TensorIterator::MergedInputDescription>(input_mapping)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);
            cldnn::primitive_id to_id = GetPrimitiveID(body_program, to);
            cldnn::primitive_id from_id = GetOutputPrimitiveID(body_program, from);
            back_edges.emplace_back(from_id, to_id);
        }
    }
    for (const auto& output_mapping : output_mappings) {
        const cldnn::primitive_id& external_id = layerName;
        const auto& body_output = body_outputs.at(output_mapping->m_body_value_index);
        const cldnn::primitive_id internal_id = GetOutputPrimitiveID(body_program, body_output);
        if (const auto& concatOutput =
            std::dynamic_pointer_cast<TensorIterator::ConcatOutputDescription>(output_mapping)) {
            // output requires concatenation
            primitive_map.emplace_back(cldnn::loop::OUTPUT, external_id, internal_id,
                concatOutput->m_axis, concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
        }
        if (const auto& body_desc =
            std::dynamic_pointer_cast<TensorIterator::BodyOutputDescription>(output_mapping)) {
            // output requires no concatenation
            primitive_map.emplace_back(cldnn::loop::OUTPUT, external_id, internal_id);
        }
    }

    const cldnn::loop loopPrimitive(layerName, inputPrimitives, body_topology,
        trip_count_id, execution_condition_id, num_iteration_id, primitive_map, back_edges);

    p.AddPrimitive(loopPrimitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, TensorIterator);

} // namespace CLDNNPlugin