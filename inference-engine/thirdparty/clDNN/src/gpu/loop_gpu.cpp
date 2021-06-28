// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "loop_inst.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "register_gpu.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace gpu {
struct loop_gpu : typed_primitive_impl<loop> {
    const loop_node& node;
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<loop_gpu>(*this);
    }

    void init_kernels() override {}

    loop_gpu(const loop_gpu& other) : typed_primitive_impl<loop>(other), node(other.node) {}
    explicit loop_gpu(const loop_node& node) : node(node) {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, loop_inst& instance) override {
        auto& outer_network = instance.get_network();
        auto& stream = outer_network.get_stream();

        auto body_network = instance.get_body_network();

        //// 아래의 이터레이션에서 인풋과 아웃풋 메모리를 바로 가져올 수 있도로 미리 계산해두는 부분.
        if (!instance.preproc_memories_done) {
            instance.preprocess_output_memory();
            instance.preprocess_input_memory();
            instance.preprocess_backedge_memory();

            // set input data for current_iteration primitive if current_iteration is used
            if (node.is_current_iteration_used()) {
                const primitive_id& current_iteration_id = node.get_current_iteration_id();
                auto current_iteration_prim = body_network->get_primitive(current_iteration_id);
                auto input_layout_prim = std::dynamic_pointer_cast<input_layout_inst>(current_iteration_prim);
                if (input_layout_prim == nullptr) {
                    CLDNN_ERROR_MESSAGE(node.id(), "current_iteration primitive is not input_layout");
                }

                const auto& backedge_mapping = instance.get_current_iteration_backedge_mapping();
                input_layout_prim->set_data(backedge_mapping.initial_mem);
            }
            instance.preproc_memories_done = true;
        }

        // read trip_count from outer network
        const primitive_id& trip_count_id = node.get_trip_count_id();
        memory::ptr trip_count_mem = outer_network.get_primitive(trip_count_id)->output_memory_ptr();
        int64_t trip_count = loop_node::read_scalar_value(trip_count_mem, stream);
        if (trip_count < 0) {
            const int64_t max_iteration = node.get_max_iteration();
            trip_count = max_iteration;
        }

        // read initial execution condition from outer network
        const primitive_id& initial_execution_id = node.get_initial_execution_id();
        memory::ptr initial_execution_mem = outer_network.get_primitive(initial_execution_id)->output_memory_ptr();
        int64_t execution_condition = loop_node::read_scalar_value(initial_execution_mem, stream);

        //// current_iteration execution_condition이 TI에서는 사용되지 않음.
        // shortcut of execution_condition memory in body network
        memory::ptr execution_condition_mem = nullptr;
        if (node.is_execution_condition_used()) {
            const primitive_id& condition_id = node.get_condition_id();
            execution_condition_mem = body_network->get_primitive(condition_id)->output_memory_ptr();
        }

        const auto& concatenated_input_mem_mappings = instance.concatenated_input_mem_mappings;
        const auto& concatenated_output_mem_mappings = instance.concatenated_output_mem_mappings;

        // Set sliced input data
        //// 바디네트웍을 실행위해서 set input data를 무조건 한번은 해주어야 함.
        for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
            const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
            memory::ptr mem = concatenated_input.get_sliced_mem(0);
            if (mem) {
                body_network->set_input_data(concatenated_input.sliced_data_prim->id(), mem);
            } else {
                CLDNN_ERROR_MESSAGE(node.id(), "sliced input memory of loop is not allocated properly");
            }
        }

        //// 이터레이션간의 이벤트들을 관리. 현재의 이벤트를 다음 익스큐션에게 넘겨주기 위해서 필요한 것
        //// 처음에는 룹이전의 노드들의 이벤트들이 저장되어 있음.
        std::vector<event::ptr> loop_carried_dep(events.begin(), events.end());
        int64_t current_iteration_idx = 0;
        while (current_iteration_idx < trip_count && execution_condition) {
            //// Question.8. loop and TI only have concat input?
            // Copy & Set sliced input memory
            //// loop input 과 body input 연결
            for (size_t i = 0; i < concatenated_input_mem_mappings.size(); ++i) {
                const auto& concatenated_input = concatenated_input_mem_mappings.at(i);
                memory::ptr mem = concatenated_input.get_sliced_mem(current_iteration_idx);
                //// iteration에 해당하는 인풋 메모리를 아웃풋 메모리에 셋팅하는 것
                //// Question.9. 이부분에서 현재의 인풋에서 아웃풋 메모리를 셋팅하면 그 다음 이터레이션의 인풋으로 연결되는 것인가?
                if (mem) {
                    //// Remove dependencies between iteration
                    //// - use set_output_memory() api to share memory between previous iteration's output and next iteration's input
                    //// - Each iteration's sliced input has its own memory
                    //// - Run body_network->execute() with the events from the previous iteration execution to ensure execution order
                    concatenated_input.sliced_data_prim->set_output_memory(mem);
                } else {
                    CLDNN_ERROR_MESSAGE(node.id(), "sliced input memory of loop is not allocated properly");
                }
            }
            /// concatenated_input_mem_mappings
            /// input     output
            ///  1          0
            ///  2          1

            // Set backedges
            /// 백엣지 메모리 셋팅
            for (const auto& backedge_memory_mapping : instance.backedge_memory_mappings) {
                backedge_memory_mapping.setup_iteration(current_iteration_idx);
            }

            // Set sliced output memory
            for (const auto& concat_output_mem_mapping : concatenated_output_mem_mappings) {
                //// 인풋과 아웃풋을 셋팅하도록 하는 함수
                concat_output_mem_mapping.setup_concatenated_output_memory(current_iteration_idx);
            }

            // execute body network
            body_network->execute(loop_carried_dep);

            loop_carried_dep.clear();
            for (const auto& backedge : node.get_back_edges()) {
                event::ptr body_event = body_network->get_primitive_event(backedge.from);
                loop_carried_dep.emplace_back(body_event);
            }

            //TODO: execution_condition is prepared as they are presented in the
            //      ngraph opset document for loop operation.
            // However they are not being used yet and only TensorIterator which
            // has fixed sequence length is being validated.
            if (node.is_execution_condition_used()) {
                execution_condition = loop_node::read_scalar_value(execution_condition_mem, stream);
            }

            // update index & execution condition for the next iteration
            ++current_iteration_idx;
        }

        body_network->reset_execution();

        // Concatenate sliced output to the outer network
        for (size_t i = 0; i < concatenated_output_mem_mappings.size(); ++i) {
            const auto& concat_output = concatenated_output_mem_mappings.at(i);
            //// 미리 계산해놓은 오프셋하고 바이트 사이즈만큼 카피하도록 해놓음.
            concat_output.restore_concatenated_mem();
        }

        // update num_iterations (actual number of iterations)
        int64_t actual_iterations = 0;
        if (node.is_current_iteration_used()) {
            const auto& backedge_mapping = instance.get_current_iteration_backedge_mapping();
            auto current_iteration_mem = backedge_mapping.from_primitive->output_memory_ptr();
            actual_iterations = loop_node::read_scalar_value(current_iteration_mem, stream);
        } else {
            actual_iterations = current_iteration_idx;
        }

        const primitive_id& num_iteration_id = node.get_num_iteration_id();
        memory::ptr num_actual_iterations_mem = outer_network.get_primitive(num_iteration_id)->output_memory_ptr();
        loop_node::write_scalar_value(num_actual_iterations_mem, stream, actual_iterations);

        return stream.create_user_event(true);
    }

    static primitive_impl* create(const loop_node& arg) { return new loop_gpu(arg); }
};

namespace detail {
attach_loop_gpu::attach_loop_gpu() {
    implementation_map<loop>::add({{engine_types::ocl, loop_gpu::create}});
}
}  // namespace detail

}  // namespace gpu
}  // namespace cldnn
