// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "softmax/softmax_kernel_selector.h"
#include "softmax/softmax_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"

namespace cldnn {
namespace gpu {

struct softmax_gpu : typed_primitive_gpu_impl<softmax> {
    using parent = typed_primitive_gpu_impl<softmax>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<softmax_gpu>(*this);
    }

    static primitive_impl* create(const softmax_node& arg) {
        auto sm_params = get_default_params<kernel_selector::softmax_params>(arg);
        auto sm_optional_params =
            get_default_optional_params<kernel_selector::softmax_optional_params>(arg);

        auto& input = sm_params.inputs[0];
        auto& output = sm_params.output;
        const auto primitive = arg.get_primitive();

        switch (primitive->dimension) {
            case softmax::normalize_x:
                sm_params.dim = kernel_selector::softmax_dim::X;
                break;

            case softmax::normalize_y:
                sm_params.dim = kernel_selector::softmax_dim::Y;
                break;

            case softmax::normalize_fyx:
                // Flatten fused with softmax
                input = input.FlattenFeatureAndSpatials();
                output = output.FlattenFeatureAndSpatials();

                sm_params.dim = kernel_selector::softmax_dim::FEATURE;
                break;

            case softmax::normalize_f:
                sm_params.dim = kernel_selector::softmax_dim::FEATURE;
                break;

            case softmax::normalize_z:
                sm_params.dim = kernel_selector::softmax_dim::Z;
                break;

            case softmax::normalize_all:
                input = input.FlattenEverything();
                output = output.FlattenEverything();

                sm_params.dim = kernel_selector::softmax_dim::FEATURE;
                break;

            default:
                throw std::runtime_error("Wrong API - no such softmax");
        }

        auto& kernel_selector = kernel_selector::softmax_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(sm_params, sm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto softmax_node = new softmax_gpu(arg, best_kernels[0]);

        return softmax_node;
    }
};

namespace detail {

attach_softmax_gpu::attach_softmax_gpu() {
    auto val_fw = softmax_gpu::create;
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<softmax>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
