// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "internal_primitive.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "program_node.h"
#include "engine_impl.h"
#include "cldnn_itt.h"

using namespace cldnn;

void compile_graph::run(program_impl& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::CompileGraph");
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>()) {
            node->get_output_layout();
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "compile_graph::run duration: " << (static_cast<double>(duration) / 1000) << "ms" << std::endl;
}
