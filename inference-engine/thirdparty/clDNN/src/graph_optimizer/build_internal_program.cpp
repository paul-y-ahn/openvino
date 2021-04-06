// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "engine_impl.h"
#include "cldnn_itt.h"
#include "loop_inst.h"

using namespace cldnn;

void build_internal_program::run(program_impl& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::BuildInternalProgram");
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<loop>()) {
            node->as<loop>().build_body_program();
        }
    }
}
