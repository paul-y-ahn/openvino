// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>
#include <functional>
#include "primitive.hpp"
#include "topology.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Adds primitive which performs recurrent execution of the topology.
///
/// @details
/// @n   The body topology for recurrent execution is described in the body
/// @n   The execution of the body topology iterates through the data in the given axis.
struct tensor_iterator : public primitive_base<tensor_iterator> {
    CLDNN_DECLARE_PRIMITIVE(tensor_iterator)

    struct primitive_mapping {
        /// @brief Constructs a mapping from external input/output primitive to input/output primitive in body topology
        ///
        /// @param external_id Primitive id of input of tensor_iterator or output of body network.
        /// @param internal_id Primitive id of input of body network.
        /// @param axis Axis to iterate through. Negative value means the axis will not iterate through and start, end, stride arguments will be ignored.
        /// @param start Index where the iteration starts from. Applies only when axis >=0.
        /// @param end Index where iteration ends. Negative value means counting indexes from the end. Applies only when axis >=0.
        /// @param stride Step of iteration. Negative value means backward iteration. Applies only when axis >=0.
        primitive_mapping(primitive_id external_id, primitive_id internal_id,
            int32_t axis = -1, int32_t start = 0, int32_t end = -1, int32_t stride = 1) :
            external_id(external_id),
            internal_id(internal_id),
            axis(axis),
            start(start),
            end(end),
            stride(stride)
            {}
        primitive_id external_id;
        primitive_id internal_id;
        int32_t axis;
        int32_t start;
        int32_t end;
        int32_t stride;
    };

    struct backedge_mapping {
        /// @brief Constructs a mapping from output of body topology to input of body topology for the next iteration
        ///
        /// @param from Output data primitive id of body topology
        /// @param to Input data primitive id of body topology
        backedge_mapping(primitive_id from, primitive_id to)
            : from(from), to(to) {}
        primitive_id from;
        primitive_id to;
    };

    /// @brief Constructs tensor_iterator primitive.
    ///
    /// @param id This primitive id.
    /// @param inputs Input data primitive id.
    /// @param body A topology to be recurrently executed.
    /// @param input_map Rules to map input of tensor_iterator or output of body topology to input of the body topology
    /// @param back_edges Output data primitive id.
    tensor_iterator(const primitive_id& id,
        const std::vector<primitive_id>& inputs,
        const topology& body,
        const std::vector<primitive_mapping>& primitive_map,
        const std::vector<backedge_mapping>& backedges,
        const padding& output_padding = padding())
            : primitive_base(id, inputs, output_padding),
              inputs(inputs),
              body(body),
              primitive_map(primitive_map),
              backedges(backedges) {}

    const std::vector<primitive_id> inputs;

    /// @brief Topology to be recurrently executed.
    // TODO: body (or body network) is the term used in TensorIterator in IR and nGraph.
    //       Should we use another name such as topology_internal?
    const topology body;

    /// @brief Rules to map input or output data of tensor_iterator layer onto input or output data of body topology.
    // TODO: The original parameter name in TensorIterator IR is port_map.
    //       cldnn does not have the term 'port' so port_map renamed as primitive_map.
    //       Should we use port_map or another name?
    const std::vector<primitive_mapping> primitive_map;

    /// @brief Rules to transfer data from body outputs at one iteration to body input at the next iteration.
    // TODO: The original parameter name in TensorIterator IR is also back edges.
    //       backedges looks self-descriptive, so is the good name for this variable.
    //       Is there another good name for this variable?
    const std::vector<backedge_mapping> backedges;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.reserve(inputs.size());
        for (const auto& input: inputs) {
            ret.push_back(std::ref(input));
        }
        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
