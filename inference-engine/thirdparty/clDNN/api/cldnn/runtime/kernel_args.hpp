// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory.hpp"

#include <memory>
#include <vector>

namespace cldnn {

struct work_group_sizes {
    std::vector<size_t> global;
    std::vector<size_t> local;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scalar_desc {
    union ValueT {
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        int8_t s8;
        int16_t s16;
        int32_t s32;
        int64_t s64;
        float f32;
        double f64;
    };

    enum class Types {
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
    };

    Types t;
    ValueT v;
};

using scalars_desc = std::vector<scalar_desc>;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ArgumentDescpirtor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct argument_desc {
    enum class Types {
        INPUT,
        OUTPUT,
        WEIGHTS,
        BIAS,
        SCALE_TABLE,
        SLOPE,
        SPLIT,
        INTERNAL_BUFFER,
        SCALAR,
        RECURRENT,  // RNN/LSTM/GRU recurrent weights
        HIDDEN,     // RNN/LSTM/GRU hidden input
        CELL,       // LSTM cell input
        LSTM_PACK,  // LSTM packed output
        WEIGHTS_ZERO_POINTS,
        ACTIVATIONS_ZERO_POINTS,
        COMPENSATION,
        INPUT_OF_FUSED_PRIMITIVE
    };

    Types t;
    uint32_t index;
};

using arguments_desc = std::vector<argument_desc>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_arguments_desc {
    work_group_sizes workGroups;
    arguments_desc arguments;
    scalars_desc scalars;
    std::string layerID;
};

struct kernel_arguments_data {
    std::vector<memory::cptr> inputs;
    std::vector<memory::cptr> intermediates;
    memory::cptr output;
    memory::cptr weights;
    memory::cptr recurrent;
    memory::cptr hidden;
    memory::cptr cell;
    memory::cptr bias;
    memory::cptr weights_zero_points;
    memory::cptr activations_zero_points;
    memory::cptr compensation;
    memory::cptr lookup_table;
    memory::cptr scale_table;
    memory::cptr slope;

    std::vector<memory::cptr> fused_op_inputs;
    int32_t split = 0;
    const scalars_desc* scalars = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelString
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_string {
    std::string str;
    std::string jit;
    std::string undefs;
    std::string options;
    std::string entry_point;
    bool batch_compilation;

    kernel_string() : str(""), jit(""), undefs(""), options(""), entry_point(""), batch_compilation(false) {}

    std::string get_str() { return str + jit + undefs + options + entry_point; }
    size_t get_hash() { return std::hash<std::string>()(get_str()); }
};

}  // namespace cldnn
