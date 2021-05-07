/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "loop_inst.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "register_gpu.hpp"
#include "mutable_data_inst.h"
#include "input_layout_inst.h"
#include "memory_impl.h"
#include <vector>
#include <algorithm>

namespace cldnn {
namespace gpu {
struct loop_gpu : typed_primitive_impl<loop> {
    const loop_node& node;

    explicit loop_gpu(const loop_node& node) : node(node) {}

    static memory_impl::ptr get_outer_output_memory(const loop_inst& instance, const primitive_id& external_id) {
        if (external_id == instance.id()) {
            // main output
            memory_impl& memory = instance.output_memory();
            return (memory_impl::ptr) &memory;
        } else {
            // other primitives
            const auto outputPrim = instance.get_network().get_primitive(external_id);
            memory_impl& memory = outputPrim->output_memory();
            return (memory_impl::ptr) &memory;
        }
    }

    std::vector<loop::primitive_mapping> find_primitive_mappings(const primitive_id& internal_id) {
        const auto& input_mappings = node.get_input_mappings();
        const auto& output_mappings = node.get_output_mappings();

        std::vector<loop::primitive_mapping> ret;
        for (const auto& pm : input_mappings) {
            if (pm.internal_id == internal_id) {
                ret.push_back(pm);
            }
        }
        for (const auto& pm : output_mappings) {
            if (pm.internal_id == internal_id) {
                ret.push_back(pm);
            }
        }

        return ret;
    }

    std::vector<memory_impl::ptr> find_cropped_mem(const primitive_id& internal_id, const loop_inst& instance) {
        const auto& input_iteration_mem = instance.input_iteration_mem;
        for (const auto& iter_info : input_iteration_mem) {
            if (iter_info.to_id == internal_id) {
                return iter_info.sliced_mem;
            }
        }
        const auto& output_iteration_mem = instance.output_iteration_mem;
        for (const auto& iter_info : output_iteration_mem) {
            if (iter_info.from_id == internal_id) {
                return iter_info.sliced_mem;
            }
        }
        return {}; // not found
    }

    void preprocess_output_memory(loop_inst& instance) {
        auto& engine = instance.get_network().get_engine();
        auto& output_iteration_mem = instance.output_iteration_mem;
        auto body_network = instance.get_body_network();
        const auto& output_primitive_map = node.get_output_mappings();
        output_iteration_mem.reserve(output_primitive_map.size());
        for (size_t i = 0; i < output_primitive_map.size(); ++i) {
            const auto& primitive_map = output_primitive_map.at(i);
            const primitive_id& external_id = primitive_map.external_id;
            const primitive_id& internal_id = primitive_map.internal_id;
            if (primitive_map.axis < 0) {
                memory_impl::ptr memory = get_outer_output_memory(instance, external_id);
                body_network->get_primitive(internal_id)->set_output_memory(*memory);
            } else {
                memory_impl::ptr to_mem = get_outer_output_memory(instance, external_id);
                auto output_prim = body_network->get_primitive(internal_id);
                layout cropped_layout = output_prim->output_memory().get_layout();

                const int max_iteration = node.get_max_iteration();
                std::vector<memory_impl::ptr> cropped_mems;
                cropped_mems.reserve(max_iteration);
                for (int j=0; j < max_iteration; ++j) {
                    memory_impl::ptr croped_mem = engine.allocate_memory(cropped_layout, 0);
                    cropped_mems.push_back(croped_mem);
                }

                const int linear_size = static_cast<int>(cropped_layout.get_linear_size());
                const int stride = linear_size * primitive_map.stride;
                const int start = primitive_map.start < 0? node.get_max_iteration() - 1: primitive_map.start;
                const int offset = linear_size * start;
                cldnn::loop_inst::sliced_memory_binding memory_binding_info(
                    primitive_map.internal_id, primitive_map.external_id,
                    to_mem, cropped_mems, linear_size, stride, offset);
                memory_binding_info.from_prim = body_network->get_primitive(internal_id);
                output_iteration_mem.push_back(memory_binding_info);
            }
        }
    }

    void preprocess_input_memory(loop_inst& instance) {
        auto& engine = instance.get_network().get_engine();
        auto& backedge_mem = instance.backedge_mem;
        auto& iteration_mem = instance.input_iteration_mem;
        auto body_network = instance.get_body_network();
        const auto& input_primitive_map = node.get_input_mappings();
        const size_t inputs_memory_count = instance.inputs_memory_count();
        for (size_t memory_num = 0; memory_num < inputs_memory_count; memory_num++) {
            const primitive_id& input_external_id = instance.dependencies().at(memory_num)->id();
            if (input_external_id == node.get_trip_count_id() ||
                input_external_id == node.get_initial_execution_id()) {
                continue;
            }
            memory_impl& memory = instance.input_memory(memory_num);
            auto input_pm_ptrs = node.find_primitive_mappings(input_external_id, input_primitive_map);
            if (input_pm_ptrs.size() == 0) {
                CLDNN_ERROR_MESSAGE(instance.id(), "loop primitive_map is incomplete");
            }
            for (size_t i = 0; i < input_pm_ptrs.size(); ++i) {
                const auto& input_pm = *input_pm_ptrs.at(i);

                // handle memory
                if (input_pm.axis >= 0) { // checks if it's a memory to iterate through
                    layout cropped_layout
                        = instance.get_body_network()->get_primitive(input_pm.internal_id)->output_memory().get_layout();
                    const int max_iteration = node.get_max_iteration();
                    std::vector<memory_impl::ptr> cropped_mems;
                    cropped_mems.reserve(max_iteration);
                    for (int j=0; j < max_iteration; ++j) {
                        memory_impl::ptr croped_mem = engine.allocate_memory(cropped_layout, 0);
                        cropped_mems.push_back(croped_mem);
                    }
                    const int linear_size = static_cast<int>(cropped_layout.get_linear_size());
                    const int stride = linear_size * input_pm.stride;
                    const int start = input_pm.start < 0? node.get_max_iteration() - 1: input_pm.start;
                    const int offset = linear_size * start;
                    loop_inst::sliced_memory_binding memory_binding_info(
                        input_pm.external_id, input_pm.internal_id,
                        (memory_impl::ptr)&memory, cropped_mems, linear_size, stride, offset);
                    memory_binding_info.to_prim = body_network->get_primitive(input_pm.internal_id);
                    iteration_mem.push_back(memory_binding_info);
                } else { // "normal" mem
                    if (memory.get_layout().data_type != body_network->get_primitive(input_pm.internal_id)->output_memory().get_layout().data_type) {
                        CLDNN_ERROR_MESSAGE(instance.id(), "incompatible datatypes");
                    }
                    body_network->set_input_data(input_pm.internal_id, memory);
                }

                // checking if memory is a destination of a backedge
                const auto& back_edges = node.get_back_edges();
                for (const auto& back_edge : back_edges) {
                    if (input_pm.internal_id != back_edge.to) {
                        continue;
                    }
                    //find corresponding input of the backedge
                    for (const auto& body_output : body_network->get_outputs()) {
                        if (body_output->id() != back_edge.from) {
                            continue;
                        }
                        const int max_iteration = node.get_max_iteration();
                        auto from_mems = find_cropped_mem(back_edge.from, instance);
                        if (from_mems.empty()) { // backedge output which does not need concatenation
                            // input memory = output memory = loop output memory
                            memory_impl::ptr loop_input_mem = get_outer_output_memory(instance, input_pm.external_id);
                            for (auto& output : body_network->get_outputs()) {
                                if (output->id() != back_edge.from) {
                                    continue;
                                }
                                const auto output_primitive_mapping = find_primitive_mappings(back_edge.from);
                                if (output_primitive_mapping.empty()) {
                                    continue;
                                }
                                memory_impl::ptr loop_output_mem = get_outer_output_memory(instance, output_primitive_mapping.front().external_id);
                                body_network->set_input_data(back_edge.to, *loop_output_mem);
                                body_network->set_output_memory(back_edge.from, *loop_output_mem);
                                copy_buffer(*loop_input_mem, *loop_output_mem, loop_input_mem->get_layout().count());
                                break;
                            }

                        } else {
                            memory_impl::ptr initial_mem = get_outer_output_memory(instance, input_pm.external_id);
                            backedge_mem.emplace_back(body_output, initial_mem);
                            for (int j = 0 ; j < max_iteration; ++j) {
                                memory_impl::ptr from_mem = from_mems.at(j);
                                backedge_mem.back().add_backedge_from_mem(from_mem);
                            }
                        }
                    }
                }
            }
        }
    }

    // extract int from data primitive
    int64_t read_int(memory_impl& mem) {
        int64_t trip_count = 0;
        const layout& prim_layout = mem.get_layout();

        switch (prim_layout.data_type) {
        case data_types::u8: {
            mem_lock<uint8_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i8: {
            mem_lock<int8_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i32: {
            mem_lock<int32_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i64: {
            mem_lock<int64_t> lock_prim_output{mem};
            trip_count = *lock_prim_output.data();
            break;
        }
        default:
            assert(false);
        }
        return trip_count;
    }

    static void copy_buffer(cldnn::memory_impl& src_mem, cldnn::memory_impl& dst_mem,
                            const size_t size, const size_t src_offset = 0, const size_t dst_offset = 0) {
        assert(src_mem.get_layout().data_type == dst_mem.get_layout().data_type);

        size_t bytes_per_element = data_type_traits::size_of(src_mem.get_layout().data_type);
        mem_lock<uint8_t> from_lock{ src_mem };
        mem_lock<uint8_t> to_lock{ dst_mem };

        const size_t byte_size_to_copy = size * bytes_per_element;
        const auto src = from_lock.begin() + src_offset * bytes_per_element;
        const auto dst = to_lock.begin() + (dst_offset * bytes_per_element);
        std::copy(src, src + byte_size_to_copy, dst);
    }

    static void copy_buffer(const primitive_id& src_id, cldnn::network_impl& src_net,
                            const primitive_id& dst_id, cldnn::network_impl& dst_net,
                            const size_t size, const size_t src_offset = 0, const size_t dst_offset = 0) {
        // TODO(cldnn loop): if not used, this should be removed
        std::shared_ptr<cldnn::primitive_inst> src_data = src_net.get_primitive(src_id);
        std::shared_ptr<cldnn::primitive_inst> dst_data = dst_net.get_primitive(dst_id);
        assert(src_data->type() == cldnn::data::type_id() || src_data->type() == cldnn::mutable_data::type_id());
        assert(dst_data->type() == cldnn::data::type_id() || dst_data->type() == cldnn::mutable_data::type_id());

        memory_impl& src_mem = src_data->output_memory();
        memory_impl& dst_mem = dst_data->output_memory();
        copy_buffer(src_mem, dst_mem, size, src_offset, dst_offset);
    }

    static void copy_entire_buffer(memory_impl& src_mem, memory_impl& dst_mem, size_t destination_offset = 0) {
        copy_buffer(src_mem, dst_mem, src_mem.get_layout().get_linear_size(), 0, destination_offset);
    }

    static void write_int(memory_impl& mem, int64_t input) {
        const layout& prim_layout = mem.get_layout();

        switch (prim_layout.data_type) {
        case data_types::u8: {
            assert(input >= std::numeric_limits<uint8_t>::min() &&
                   input <= std::numeric_limits<uint8_t>::max());
            mem_lock<uint8_t> lock_prim_output{mem};
            *lock_prim_output.data() = static_cast<uint8_t>(input);
            break;
        }
        case data_types::i8: {
            assert(input >= std::numeric_limits<int8_t>::min() &&
                   input <= std::numeric_limits<int8_t>::max());
            mem_lock<int8_t> lock_prim_output{mem};
            *lock_prim_output.data() = static_cast<int8_t>(input);
            break;
        }
        case data_types::i32: {
            assert(input >= std::numeric_limits<int32_t>::min() &&
                   input <= std::numeric_limits<int32_t>::max());
            mem_lock<int32_t> lock_prim_output{mem};
            *lock_prim_output.data() = static_cast<int32_t>(input);
            break;
        }
        case data_types::i64: {
            mem_lock<int64_t> lock_prim_output{mem};
            *lock_prim_output.data() = input;
            break;
        }
        default:
            assert(false);
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, loop_inst& instance) override {
        for (auto& e : events)
            e->wait();
        auto& outer_network = instance.get_network();
        const uint32_t& net_id = instance.get_network().get_id();
        auto ev = outer_network.get_engine().create_user_event(net_id, false);

        auto body_network = instance.get_body_network();

        // read trip_count from outer network
        const primitive_id& trip_count_id = node.get_trip_count_id();
        memory_impl& trip_count_mem = outer_network.get_primitive(trip_count_id)->output_memory();
        int64_t trip_count = read_int(trip_count_mem);
        if (trip_count < 0) {
            trip_count = std::numeric_limits<int64_t>::max(); // infinity loop
        }

        // read initial execution condition from outer network
        const primitive_id& initial_execution_id = node.get_initial_execution_id();
        memory_impl& initial_execution_mem = outer_network.get_primitive(initial_execution_id)->output_memory();
        int64_t execution_condition = read_int(initial_execution_mem);

        // shortcut of current_iteration memory in body network (slice of input)
        memory_impl* current_iteration_mem = nullptr;
        if (node.is_current_iteration_used()) {
            const primitive_id& current_iteration_id = node.get_current_iteration_id();
            current_iteration_mem = &body_network->get_primitive(current_iteration_id)->output_memory();
        }


        // shortcut of execution_condition memory in body network
        memory_impl* execution_condition_mem = nullptr;
        if (node.is_execution_condition_used()) {
            const primitive_id& condition_id = node.get_condition_id();
            execution_condition_mem = &body_network->get_primitive(condition_id)->output_memory();
        }

        // output memory must be set before input_memory to set backedge memory properly
        if (!instance.memroy_set) {
            preprocess_output_memory(instance);
            preprocess_input_memory(instance);
            instance.memroy_set = true;
        }

        int64_t current_iteration = 0;
        if (node.is_current_iteration_used()) {
            write_int(*current_iteration_mem, current_iteration);
        }
        const auto& input_iteration_mem = instance.input_iteration_mem;
        const auto& output_iteration_mem = instance.output_iteration_mem;
        int actual_iteration = 0;
        std::vector<event_impl::ptr> body_events;
        while (current_iteration < trip_count && execution_condition) {
            // Copy & Set sliced input memory offset
            for (size_t i = 0; i < instance.input_iteration_mem.size(); ++i) {
                const auto& cropped_mem_info = input_iteration_mem.at(i);
                memory_impl::ptr mem = cropped_mem_info.copy_and_get_sliced_mem(actual_iteration);
                // set input mem
                if (actual_iteration == 0) {
                    body_network->set_input_data(cropped_mem_info.to_id, *mem);
                } else {
                    cropped_mem_info.to_prim->set_output_memory(*mem);
                }
            }

            // Set backedged input
            if (actual_iteration == 0) {
                for (auto& edge_mem_bind : instance.backedge_mem) {
                    const primitive_id& input_id = edge_mem_bind.to_primitive->id();
                    body_network->set_input_data(input_id, *edge_mem_bind.initial_mem);
                }
            } else {
                for (auto& edge_mem_bind : instance.backedge_mem) {
                    edge_mem_bind.set_backedged_input(actual_iteration);
                }
            }

            // Set sliced output memory offset
            for (size_t i = 0; i < output_iteration_mem.size(); ++i) {
                const auto& cropped_mem_info = output_iteration_mem.at(i);
                const auto& from_mem = cropped_mem_info.sliced_mem.at(actual_iteration);
                cropped_mem_info.from_prim->set_output_memory(*from_mem);
            }
            if (actual_iteration == 0) {
                body_network->execute(events);
            } else {
                body_events.clear();
                for (const auto& backedge : node.get_back_edges()) {
                    event_impl::ptr body_event = body_network->get_primitive_event(backedge.from);
                    body_events.emplace_back(body_event);
                }
                body_network->execute(body_events);
            }


            // update index & execution condition for the next iteration
            if (node.is_current_iteration_used()) {
                current_iteration = read_int(*current_iteration_mem);
                ++current_iteration;
                write_int(*current_iteration_mem, current_iteration);
            } else {
                ++current_iteration;
            }
            if (node.is_execution_condition_used()) {
                execution_condition = read_int(*execution_condition_mem);
            }
            ++actual_iteration;
        }
        body_network->reset_execution();

        // Concatenate sliced output to the outer network
        for (size_t i = 0; i < output_iteration_mem.size(); ++i) {
            const auto& cropped_mem_info = output_iteration_mem.at(i);
            cropped_mem_info.copy_to_concatenated_mem();
        }

        const primitive_id& num_iteration_id = node.get_num_iteration_id();
        memory_impl& num_iteration_mem = outer_network.get_primitive(num_iteration_id)->output_memory();
        write_int(num_iteration_mem, actual_iteration);

        dynamic_cast<cldnn::user_event*>(ev.get())->set();
        return ev;
    }

    static primitive_impl* create(const loop_node& arg) { return new loop_gpu(arg); }
};

namespace detail {
attach_loop_gpu::attach_loop_gpu() {
    std::vector<data_types> loop_data_types{ data_types::bin, data_types::u8, data_types::i8, data_types::f16,
                                             data_types::f32, data_types::i32, data_types::i64};

    std::vector<format> loop_formats{ format::bfyx, format::bfzyx, format::bfwzyx };

    for (const data_types loop_data_type : loop_data_types) {
        for (const format loop_format : loop_formats) {
            implementation_map<loop>::add(
                std::make_tuple(engine_types::ocl, loop_data_type, loop_format),
                loop_gpu::create);
        }
    }
}
}  // namespace detail

}  // namespace gpu
}  // namespace cldnn
