#include <torch/script.h>  // One-stop header.

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataProcessing.hpp"

using namespace cv;
using namespace std::chrono;

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
    PostProcessing detection("params.txt");
    PreProcessing preprocess("params.txt");
    std::string path =
        "/home/fabian/data/TS/ImageTCL/grayscale/"
        "2000-01-01_109790322_121528_REF_003754.png";
    for (int i = 0; i < 10; ++i) {
        cv::Mat image;
        image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
        torch::Tensor tensor_image = preprocess.process(image);
        auto start = high_resolution_clock::now();
        auto outputs = module.forward(inputs).toTuple();
        auto final = detection.process(outputs->elements()[0].toTensor(),
                                       outputs->elements()[1].toTensor(),
                                       priors.attr("priors").toTensor());
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Time taken by function: at " << i << " "
                  << duration.count() << " microseconds" << std::endl;
    }
}
