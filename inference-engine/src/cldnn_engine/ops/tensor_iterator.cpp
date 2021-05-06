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
#include "api/data.hpp"
#include "api/reorder.hpp"

#include <vector>
#include <algorithm>

using TensorIterator = ngraph::op::v0::TensorIterator;

namespace CLDNNPlugin {

template<class DATA_TYPE>
static DATA_TYPE CreateIntData(Program &p, const cldnn::primitive_id& id, int32_t num) {
    auto mem = cldnn::memory::allocate(p.GetEngine(),
        { cldnn::data_types::i32, cldnn::format::bfyx, { 1, 1, 1, 1 } });
    auto ptr = mem.pointer<int32_t>();
    *ptr.begin() = num;
    return {id, mem};
}

static cldnn::mutable_data CreateOutputMutableData(Program &p, const std::shared_ptr<ngraph::Node>& op,
                                            const cldnn::primitive_id& id, const cldnn::primitive_id& input,
                                            const int32_t output_idx) {
    const auto precision = DataTypeFromPrecision(op->get_output_element_type(output_idx));
    const auto format = DefaultFormatForDims(op->get_output_shape(output_idx).size());
    const auto tensor = CldnnTensorFromIEDims(op->get_output_shape(output_idx));
    cldnn::layout output_layout = cldnn::layout(precision, format, tensor);
    auto mem = cldnn::memory::allocate(p.GetEngine(), output_layout);
    auto md = cldnn::mutable_data(id, {input}, mem); // cldnn::data cannot set dependency
    return md;
}

static void UpdateBackedge(std::vector<cldnn::loop::backedge_mapping>& back_edges,
        const cldnn::primitive_id& old_primitive_id, const cldnn::primitive_id& new_primitive_id) {
    for (auto& back_edge : back_edges) {
        if (back_edge.from == old_primitive_id) {
            back_edge.from = new_primitive_id;
        }
    }
}

void CreateTensorIteratorOp(Program &p, const std::shared_ptr<TensorIterator> &op) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    // get body topology from ngraph function
    InferenceEngine::CNNNetwork body_network(op->get_body());
    Program body_program(body_network, p.GetEnginePtr(), p.GetConfig(), false);
    auto body_topology = *body_program.GetTopology();

    // setup input_mappings/ output_mappings and back_edges
    const auto& loop_input_descs = op->get_input_descriptions();
    const auto& loop_output_descs = op->get_output_descriptions();
    const auto& body_inputs = op->get_body()->get_parameters();
    const auto& body_outputs = op->get_body()->get_results();

    std::vector<cldnn::loop::primitive_mapping> input_mappings;
    std::vector<cldnn::loop::primitive_mapping> output_mappings;
    std::vector<cldnn::loop::backedge_mapping> back_edges;

    // set input mapping & back edges
    for (const auto& loop_input_desc : loop_input_descs) {
        const cldnn::primitive_id& external_id = inputPrimitives.at(loop_input_desc->m_input_index);
        auto& body_input = body_inputs.at(loop_input_desc->m_body_parameter_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_input);

        // set input mapping
        if (const auto& sliceInfo =
            std::dynamic_pointer_cast<TensorIterator::SliceInputDescription>(loop_input_desc)) {
            // sliced input
            input_mappings.emplace_back(external_id, internal_id, sliceInfo->m_axis,
                sliceInfo->m_start, sliceInfo->m_end, sliceInfo->m_stride);
        } else {
            // input without slicing
            input_mappings.emplace_back(external_id, internal_id);
        }

        // set back edges
        if (const auto& mergedInput =
            std::dynamic_pointer_cast<TensorIterator::MergedInputDescription>(loop_input_desc)) {
            // backedge
            const auto& to = body_inputs.at(mergedInput->m_body_parameter_index);
            const auto& from = body_outputs.at(mergedInput->m_body_value_index);
            cldnn::primitive_id to_id = layer_type_name_ID(to);
            cldnn::primitive_id from_id = layer_type_name_ID(from);
            back_edges.emplace_back(from_id, to_id);
        }
    }

    // set trip count, initial execution condition, num iteration primitives
    // they should be mutable_data to prevent from being optimized out
    std::string layerName = layer_type_name_ID(op);
    const cldnn::primitive_id trip_count_id = layerName + "_tripCount";
    const int64_t num_iterations = op->get_num_iterations();
    assert(num_iterations >= 0);
    {
        cldnn::data trip_count = CreateIntData<cldnn::data>(p, trip_count_id, num_iterations);
        p.primitivesToIRLayersMap[trip_count_id] = { op->get_friendly_name() };
        p.primitiveIDs[trip_count_id] = trip_count_id;
        p.AddPrimitive(trip_count);
        p.AddInnerPrimitiveToProfiler(trip_count_id, layerName, op);
    }
    const cldnn::primitive_id execution_condition_id = layerName + "_initialExecutionCondition";
    {
        cldnn::mutable_data execution_condition = CreateIntData<cldnn::mutable_data>(p, execution_condition_id, 1);
        p.primitivesToIRLayersMap[execution_condition_id] = { op->get_friendly_name() };
        p.primitiveIDs[execution_condition_id] = execution_condition_id;
        p.AddPrimitive(execution_condition);
        p.AddInnerPrimitiveToProfiler(execution_condition_id, layerName, op);
    }
    const cldnn::primitive_id num_iteration_id = layerName + "_numIteration";
    {
        cldnn::mutable_data num_iteration = CreateIntData<cldnn::mutable_data>(p, num_iteration_id, 0);
        p.primitivesToIRLayersMap[num_iteration_id] = { op->get_friendly_name() };
        p.primitiveIDs[num_iteration_id] = num_iteration_id;
        p.AddPrimitive(num_iteration);
        p.AddInnerPrimitiveToProfiler(num_iteration_id, layerName, op);
    }

    // set output mapping
    const auto& ti_outputs = op->outputs();
    for (const auto& loop_output_desc : loop_output_descs) {
        const int output_idx = loop_output_desc->m_output_index;

        // Add additional mutable_data for multiple outputs
        // primitive ID should be <TI primitive ID>.<output_idx> if output_idx > 0
        // otherwise primitive ID should be equals to TI primitive ID
        const std::string layerNameWithIndex = layerName + "." + std::to_string(output_idx);
        std::string external_id;
        if (output_idx > 0) {
            cldnn::mutable_data output_data = CreateOutputMutableData(p, op, layerNameWithIndex, layerName, output_idx);
            p.primitiveIDs[layerNameWithIndex] = layerNameWithIndex;
            p.AddPrimitive(output_data);
            p.AddInnerPrimitiveToProfiler(layerNameWithIndex, layerName, op);
            external_id = layerNameWithIndex;
        } else {
            p.primitiveIDs[layerNameWithIndex] = layerName;
            p.primitiveIDs[layerName] = layerName;
            external_id = layerName;
        }
        const auto& body_output = body_outputs.at(loop_output_desc->m_body_value_index);
        cldnn::primitive_id internal_id = layer_type_name_ID(body_output);

        // TODO(eunsoo): reorder required?
        // add additional reorder in case TI output type != body output type
        const auto& ti_output_type = ti_outputs.at(output_idx).get_element_type();
        cldnn::primitive_id new_internal_id = internal_id + "_reorder";
        const auto new_body_output_type = DataTypeFromPrecision(ti_output_type);
        auto reorderPrim = cldnn::reorder(new_internal_id, internal_id, cldnn::format::any, new_body_output_type);
        body_topology.add(reorderPrim);
        UpdateBackedge(back_edges, internal_id, new_internal_id);
        internal_id = std::move(new_internal_id);

        // update primitive_map
        if (const auto& concatOutput =
            std::dynamic_pointer_cast<TensorIterator::ConcatOutputDescription>(loop_output_desc)) {
            // output which requires concatenation
            output_mappings.emplace_back(external_id, internal_id, concatOutput->m_axis,
                concatOutput->m_start, concatOutput->m_end, concatOutput->m_stride);
        }
        if (const auto& body_desc =
            std::dynamic_pointer_cast<TensorIterator::BodyOutputDescription>(loop_output_desc)) {
            // output which requires no concatenation
            output_mappings.emplace_back(external_id, internal_id);
        }
    }

    const cldnn::loop loopPrimitive(
        layerName,              /* layer name of this primitive (output id) */
        inputPrimitives,        /* inputs of this layer */
        body_topology,          /* body network */
        trip_count_id,          /* trip_count data in outer network, always same as num_iterations in TI */
        execution_condition_id, /* initial_execution_condition data in outer network, always true in TI */
        num_iteration_id,       /* actual number of iteration data in body network */
        input_mappings,         /* input mappings connecting outer network and inner network */
        output_mappings,        /* output mappings connecting outer network and inner network */
        back_edges,             /* back edge mapping */
        num_iterations);        /* max iteration, i.e. length of iteration axis */

    p.AddPrimitive(loopPrimitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, TensorIterator);

} // namespace CLDNNPlugin
