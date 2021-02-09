// Copyright (c) 2020 Intel Corporation
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


#include "include/include_all.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif

#if OUTPUT_DIMS != INPUT2_DIMS
    #error "OUTPUT_DIMS is supposed to be same as INPUT2_DIMS"
#endif

KERNEL(scatter_nd_update_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates, 
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

#ifndef IS_SECOND_ITER // First kernel
    // Find rank
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel
    printf("INPUT0_SIZE_Z:%d, INPUT0_SIZE_Y:%d, INPUT0_SIZE_X:%d \n", INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X);

    // Kelvin: I'm not sure why the position of X and Y is reverted!
    #if INPUT0_SIZE_Y != 1
        #define DATA_RANK           (INPUT0_DIMS - 0)
    #elif INPUT0_SIZE_X != 1
        #define DATA_RANK           (INPUT0_DIMS - 1)
    #elif INPUT0_SIZE_Z != 1
        #define DATA_RANK           (INPUT0_DIMS - 2)
    #elif INPUT0_SIZE_W != 1
        #define DATA_RANK           3
    #elif INPUT0_FEATURE_NUM != 1
        #define DATA_RANK           2
    #else // INPUT0_BATCH_NUM != 1
        #define DATA_RANK           1
    #endif

    #if INPUT1_SIZE_Y != 1
        #define INDICE_RANK         (INPUT1_DIMS - 0)
    #elif INPUT1_SIZE_X != 1
        #define INDICE_RANK         (INPUT1_DIMS - 1)
    #elif INPUT1_SIZE_Z != 1
        #define INDICE_RANK         (INPUT1_DIMS - 2)
    #elif INPUT1_SIZE_W != 1
        #define INDICE_RANK         3
    #elif INPUT1_FEATURE_NUM != 1
        #define INDICE_RANK         2
    #else // INPUT1_BATCH_NUM != 1
        #define INDICE_RANK         1
    #endif

    #if INPUT2_SIZE_Y != 1
        #define UPDATE_RANK         (INPUT2_DIMS - 0)
    #elif INPUT2_SIZE_X != 1
        #define UPDATE_RANK         (INPUT2_DIMS - 1)
    #elif INPUT2_SIZE_Z != 1
        #define UPDATE_RANK         (INPUT2_DIMS - 2)
    #elif INPUT2_SIZE_W != 1
        #define UPDATE_RANK         3
    #elif INPUT2_FEATURE_NUM != 1
        #define UPDATE_RANK         2
    #else // INPUT2_BATCH_NUM != 1
        #define UPDATE_RANK         1
    #endif


    printf("DATA_RANK:%d, INDICE_RANK:%d, UPDATE_RANK:%d \n", DATA_RANK, INDICE_RANK, UPDATE_RANK);
        

    // item by indice order(INPUT1)
    #if OUTPUT_DIMS == 4
        const uint idx_x = dim0;
        const uint idx_y = dim1;
        const uint idx_f = dim2 % INPUT1_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT1_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint idx_x = dim0 % INPUT1_SIZE_X;
        const uint idx_y = dim0 / INPUT1_SIZE_X;
        const uint idx_z = dim1;
        const uint idx_f = dim2 % INPUT1_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT1_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint idx_x = dim0 % INPUT1_SIZE_X;
        const uint idx_y = dim0 / INPUT1_SIZE_X;
        const uint idx_z = dim1 % INPUT1_SIZE_Z;
        const uint idx_w = dim1 / INPUT1_SIZE_Z;
        const uint idx_f = dim2 % INPUT1_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT1_FEATURE_NUM;
    #endif

    printf("dim0:%d, dim1:%d, dim2:%d \n", dim0, dim1, dim2);
    printf("idx_b:%d, idx_f:%d, idx_y:%d, idx_x:%d \n",idx_b, idx_f, idx_y, idx_x);
   

    uint indices_idx = GET_UPDATES_INDEX(INPUT1, IDX_ORDER);
    uint output_idx = 0;
    uint update_idx = 0;
    uint update_len = 1;


    printf("Case data_rank=%d, indice_rank=1 \n", DATA_RANK, INDICE_RANK);
    #if (INDICE_RANK == 1)
        const uint x = idx_x; const uint y = idx_y; const uint f = idx_f; const uint b = indices[idx_b];
        output_idx = GET_UPDATES_INDEX(INPUT0, ORDER);
        update_idx = GET_UPDATES_INDEX(INPUT2, IDX_ORDER);
        update_len = INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X* INPUT0_SIZE_Z* INPUT0_SIZE_W;
    #else
        // Kelvin
        printf("!!! Not implemented case: This should be not worked !!! \n");
    #endif

    printf("output_idx:%d\n", output_idx);
    printf("update_idx:%d\n", update_idx);
    printf("update_len:%d\n", update_len);

    for (int i = 0; i < update_len; i++) {
        INPUT2_TYPE val = updates[update_idx + i];
        output[output_idx + i] = ACTIVATION(val, ACTIVATION_PARAMS);
    }
#endif
}

#undef DATA_RANK
#undef INDICE_RANK
#undef UPDATE_RANK

#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef IDX_ORDER
#undef ORDER
