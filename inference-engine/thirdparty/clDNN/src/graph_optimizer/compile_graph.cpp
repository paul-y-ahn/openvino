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
#include "../src/gpu/ocl_toolkit.h"
#include <iostream>

#define CLDNN_THREADING_SEQ 0
#define CLDNN_THREADING_TBB 1
#define CLDNN_THREADING_THREADPOOL 2

#ifndef CLDNN_THREADING
#define CLDNN_THREADING CLDNN_THREADING_TBB
#endif

#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#endif

#if (CLDNN_THREADING != CLDNN_THREADING_SEQ)
#define DEFAULT_NUM_THREADS 2
#endif
using namespace cldnn;

void compile_graph::run(program_impl& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::CompileGraph");
    size_t count = 0;
    size_t real_count = 0;
    auto start = std::chrono::high_resolution_clock::now();
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
    const auto n_threads = p.get_engine().get_context()->get_configuration().n_threads;
    // std::cout << "CLDNN_THREADING_TBB - n_threads: " << n_threads << std::endl;
    auto arena = std::unique_ptr<tbb::task_arena>(new tbb::task_arena());
    arena->initialize(n_threads);
    auto& proc_order = p.get_processing_order();
    arena->execute([this, &proc_order, &p, &count, &real_count] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, proc_order.size()), [&proc_order, &p, &count, &real_count](const tbb::blocked_range<size_t>& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                count++;
                auto& node = *(std::next(proc_order.begin(), i));
                node->set_unique_id(i);
                if (!node->is_type<internal_primitive>() && !node->is_type<data>()) {
                    node->get_output_layout();
                    if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                        real_count++;
                        node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
                    }
                }
            }
        });
    });
    arena.reset();
#elif(CLDNN_THREADING == CLDNN_THREADING_THREADPOOL)
    const auto n_threads = p.get_engine().get_context()->get_configuration().n_threads;
    // std::cout << "CLDNN_THREADING_THREADPOOL - n_threads: " << n_threads << std::endl;
    auto pool = std::unique_ptr<thread_pool>(new thread_pool(n_threads));
    std::vector<std::future<void>> builds;
    for (auto& node : p.get_processing_order()) {
        builds.push_back(pool->enqueue([this, &node, p] () {
            auto idx = p.get_processing_order().get_processing_number(node) - 1;
            node->set_unique_id(idx);
            if (!node->is_type<internal_primitive>() && !node->is_type<data>()) {
                node->get_output_layout();
                if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                    node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
                }
            }
        }));
    }
    std::for_each(builds.begin(), builds.end(), [] (std::future<void>& f) { f.wait(); });
    pool.reset();
#else
    // std::cout << "CLDNN_THREADING_SEQ" << std::endl;
    for (auto& node : p.get_processing_order()) {
        count++;
        auto idx = p.get_processing_order().get_processing_number(node) - 1;
        node->set_unique_id(idx);
        if (!node->is_type<internal_primitive>() && !node->is_type<data>()) {
            node->get_output_layout();
            if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
                real_count++;
                node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
            }
        }
    }
#endif
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "compile_graph::run duration[" << real_count << "/" << count << "]: " << (static_cast<double>(duration) / 1000) << "ms" << std::endl;
}
