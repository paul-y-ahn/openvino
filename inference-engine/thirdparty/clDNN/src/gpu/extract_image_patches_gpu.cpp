// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"

#include "extract_image_patches/extract_image_patches_kernel_selector.h"
#include "extract_image_patches/extract_image_patches_kernel_ref.h"

namespace cldnn {
namespace gpu {

struct extract_image_patches_gpu : typed_primitive_gpu_impl<extract_image_patches> {
    using parent = typed_primitive_gpu_impl<extract_image_patches>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<extract_image_patches_gpu>(*this);
    }

public:
    static primitive_impl* create(const extract_image_patches_node& arg) {
        auto params = get_default_params<kernel_selector::extract_image_patches_params>(arg);
        auto optional_params =
            get_default_optional_params<kernel_selector::extract_image_patches_optional_params>(arg);

        params.sizes = arg.get_primitive()->sizes;
        params.strides = arg.get_primitive()->strides;
        params.rates = arg.get_primitive()->rates;
        params.auto_pad = arg.get_primitive()->auto_pad;

        auto& kernel_selector = kernel_selector::extract_image_patches_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto extract_image_patches = new extract_image_patches_gpu(arg, best_kernels[0]);

        return extract_image_patches;
    }
};

namespace detail {

attach_extract_image_patches_gpu::attach_extract_image_patches_gpu() {
    implementation_map<extract_image_patches>::add(
        {{std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), extract_image_patches_gpu::create},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), extract_image_patches_gpu::create}});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
