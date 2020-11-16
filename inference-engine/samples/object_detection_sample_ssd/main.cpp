// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>

#include <format_reader_ptr.h>
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <vpu/vpu_tools_common.hpp>
#include <vpu/vpu_plugin_config.hpp>

#include "object_detection_sample_ssd.h"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Input is required but not set. Please set -i option.");
    }

    return true;
}

static std::map<std::string, std::string> configure(const std::string& confFileName) {
    auto config = parseConfig(confFileName);

    return config;
}

/**
* \brief The entry point for the Inference Engine object_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        /** This sample covers certain topology and cannot be generalized for any object detection one **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";

        // --------------------------- 1. Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read input -----------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            ie.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }

        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } }, "GPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // 4. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files:"
            "\n\t" << FLAGS_m <<
            slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Prepare input blobs --------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputsInfo(network.getInputsInfo());

        /** SSD network has one input and one output **/
        if (inputsInfo.size() != 1 && inputsInfo.size() != 2) throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

        /**
         * Some networks have SSD-like output format (ending with DetectionOutput layer), but
         * having 2 inputs as Faster-RCNN: one for image and one for "image info".
         *
         * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
         * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
         */
        std::string imageInputName, imInfoInputName;

        InputInfo::Ptr inputInfo = nullptr;

        SizeVector inputImageDims;
        /** Stores input image **/

        /** Iterating over all input blobs **/
        for (auto & item : inputsInfo) {
            /** Working with first input tensor that stores image **/
            if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
                imageInputName = item.first;

                inputInfo = item.second;

                slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

                /** Creating first input blob **/
                Precision inputPrecision = Precision::U8;
                item.second->setPrecision(inputPrecision);
            } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
                imInfoInputName = item.first;

                Precision inputPrecision = Precision::FP32;
                item.second->setPrecision(inputPrecision);
                if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6)) {
                    throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
                }
            }
        }

        if (inputInfo == nullptr) {
            inputInfo = inputsInfo.begin()->second;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputsInfo(network.getOutputsInfo());

        std::string outputName;
        DataPtr outputInfo;
        if (auto ngraphFunction = network.getFunction()) {
            for (const auto& out : outputsInfo) {
                for (const auto & op : ngraphFunction->get_ops()) {
                    if (op->get_type_info() == ngraph::op::DetectionOutput::type_info &&
                            op->get_friendly_name() == out.second->getName()) {
                        outputName = out.first;
                        outputInfo = out.second;
                        break;
                    }
                }
            }
        } else {
            outputInfo = outputsInfo.begin()->second;
            outputName = outputInfo->getName();
        }

        if (outputInfo == nullptr) {
            throw std::logic_error("Can't find a DetectionOutput layer in the topology");
        }

        const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

        const int maxProposalCount = outputDims[2];
        const int objectSize = outputDims[3];

        if (objectSize != 7) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }

        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD model");
        }

        /** Set the precision of output data provided by the user, should be called before load of the network to the device **/
        outputInfo->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;

        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d, configure(FLAGS_config));
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 9. Prepare input --------------------------------------------------------
        /** Collect images data ptrs **/
        std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
        std::vector<size_t> imageWidths, imageHeights;
        for (auto & i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> originalData(reader->getData());
            std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                originalImagesData.push_back(originalData);
                imagesData.push_back(data);
                imageWidths.push_back(reader->width());
                imageHeights.push_back(reader->height());
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;
        if (batchSize != imagesData.size()) {
            slog::warn << "Number of images " + std::to_string(imagesData.size()) + \
                " doesn't match batch size " + std::to_string(batchSize) << slog::endl;
            batchSize = std::min(batchSize, imagesData.size());
            slog::warn << "Number of images to be processed is "<< std::to_string(batchSize) << slog::endl;
        }

        /** Creating input blob **/
        Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

        /** Filling input tensor with images. First b channel, then g and r channels **/
        MemoryBlob::Ptr mimage = as<MemoryBlob>(imageInput);
        if (!mimage) {
            slog::err << "We expect image blob to be inherited from MemoryBlob, but by fact we were not able "
                "to cast imageInput to MemoryBlob" << slog::endl;
            return 1;
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto minputHolder = mimage->wmap();

        size_t num_channels = mimage->getTensorDesc().getDims()[1];
        size_t image_size = mimage->getTensorDesc().getDims()[3] * mimage->getTensorDesc().getDims()[2];

        unsigned char *data = minputHolder.as<unsigned char *>();

        /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_channels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                }
            }
        }

        if (imInfoInputName != "") {
            Blob::Ptr input2 = infer_request.GetBlob(imInfoInputName);
            auto imInfoDim = inputsInfo.find(imInfoInputName)->second->getTensorDesc().getDims()[1];

            /** Fill input tensor with values **/
            MemoryBlob::Ptr minput2 = as<MemoryBlob>(input2);
            if (!minput2) {
                slog::err << "We expect input2 blob to be inherited from MemoryBlob, but by fact we were not able "
                    "to cast input2 to MemoryBlob" << slog::endl;
                return 1;
            }
            // locked memory holder should be alive all time while access to its buffer happens
            auto minput2Holder = minput2->wmap();
            float *p = minput2Holder.as<PrecisionTrait<Precision::FP32>::value_type *>();

            for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
                p[image_id * imInfoDim + 0] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[2]);
                p[image_id * imInfoDim + 1] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[3]);
                for (size_t k = 2; k < imInfoDim; k++) {
                    p[image_id * imInfoDim + k] = 1.0f;  // all scale factors are set to 1.0
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 10. Do inference ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 11. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
        MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast output to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto moutputHolder = moutput->rmap();
        const float *detection = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

        std::vector<std::vector<int> > boxes(batchSize);
        std::vector<std::vector<int> > classes(batchSize);

        /* Each detection has image_id that denotes processed image */
        for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
            auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
            if (image_id < 0) {
                break;
            }

            float confidence = detection[curProposal * objectSize + 2];
            auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
            auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * imageWidths[image_id]);
            auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * imageHeights[image_id]);
            auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * imageWidths[image_id]);
            auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * imageHeights[image_id]);

            std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
                "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

            if (confidence > 0.5) {
                /** Drawing only objects with >50% probability **/
                classes[image_id].push_back(label);
                boxes[image_id].push_back(xmin);
                boxes[image_id].push_back(ymin);
                boxes[image_id].push_back(xmax - xmin);
                boxes[image_id].push_back(ymax - ymin);
                std::cout << " WILL BE PRINTED!";
            }
            std::cout << std::endl;
        }

        for (size_t batch_id = 0; batch_id < batchSize; ++batch_id) {
            addRectangles(originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id], boxes[batch_id], classes[batch_id],
                          BBOX_THICKNESS);
            const std::string image_path = "out_" + std::to_string(batch_id) + ".bmp";
            if (writeOutputBmp(image_path, originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id])) {
                slog::info << "Image " + image_path + " created!" << slog::endl;
            } else {
                throw std::logic_error(std::string("Can't create a file: ") + image_path);
            }
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}
