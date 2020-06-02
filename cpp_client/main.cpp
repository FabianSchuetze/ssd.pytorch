#include <ATen/core/ivalue.h>
#include <torch/script.h>  // One-stop header.

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataProcessing.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std::chrono;

std::vector<std::string> load_images(const std::string& path) {
    std::vector<std::string> files;
    for (const auto & img : fs::directory_iterator(path)) {
        files.push_back(img.path());
    }
    return files;
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    torch::jit::script::Module priors;
    try {
        module = torch::jit::load(argv[1]);
        priors = torch::jit::load(argv[2]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::vector<torch::jit::IValue> inputs(1);
    std::string config =
        "/home/fabian/Documents/work/github/ssd.pytorch/cpp_client/params.txt";
    PostProcessing detection(config);
    PreProcessing preprocess(config);
    std::string path = "/home/fabian/data/TS/ImageTCL/grayscale/";
    std::vector<std::string> files = load_images(path);
    for (const std::string& img : files) {
        cv::Mat image;
        image = cv::imread(img, CV_LOAD_IMAGE_COLOR);
        torch::Tensor tensor_image = preprocess.process(image);
        // torch::IValue test {image};
        inputs[0] = tensor_image;
        auto start = high_resolution_clock::now();
        auto outputs = module.forward(inputs).toTuple();
        torch::Tensor result = detection.process(outputs->elements()[0].toTensor(),
                                       outputs->elements()[1].toTensor(),
                                       priors.attr("priors").toTensor());
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Time taken by function: at " << img << " "
                  << duration.count() << " microseconds" << std::endl;
    }
}
