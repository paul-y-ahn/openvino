# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(openvino_fuzzing)

cmake_minimum_required(VERSION 3.13)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

if (NOT TARGET IE::inference_engine)
    find_package(IEDevScripts REQUIRED
                 PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
                 NO_CMAKE_FIND_ROOT_PATH
                 NO_DEFAULT_PATH)
endif()

find_package(InferenceEngine REQUIRED)

if (NOT ENABLE_FUZZING)
    message(STATUS
        "Fuzz tests will be built without fuzzer support. You can use those to\n"
        "run crash reproducers and corpus inputs. Configure ENABLE_FUZZING=ON\n"
        "to built with a fuzzer.")
else()
    enable_fuzzing()
endif()

add_subdirectory(fuzz-testhelper)
add_subdirectory(src)
