#include <ATen/core/ivalue.h>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

#include "DataProcessing.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std::chrono;

std::vector<std::string> load_images(const std::string& path) {
    std::vector<std::string> files;
    for (const auto& img : fs::directory_iterator(path)) {
        files.push_back(img.path());
    }
    return files;
}

void serialize_results(const std::string& file,
                       const std::vector<PostProcessing::Landmark>& result) {
    std::ofstream myfile;
    size_t pos = file.find_last_of("/");
    std::string filename = file.substr(pos + 1);
    std::cout << "the filename is: " << filename << std::endl;
    std::string token = filename.substr(0, filename.find("."));
    myfile.open("results/" + token + ".result", std::ios::trunc);
    std::cout << "iterating over results, with size " <<  result.size() << std::endl;
    for (const PostProcessing::Landmark& res : result) {
        myfile << "top :" << res.top << ", " << res.left << ", " <<
            res.width << ", " << res.height << "; " << res.confidence <<
            std::endl;
    }
    myfile.close();
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
    std::string path =
        "/home/fabian/data/TS/CrossCalibration/ImageTCL/greyscale/";
    std::vector<std::string> files = load_images(path);
    for (const std::string& img : files) {
        cv::Mat image, image_rgb;
        image = cv::imread(img, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(image, image_rgb, COLOR_BGR2RGB);
        //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        //imshow( "Display window", image_rgb);                   // Show our image inside it.
        //waitKey(0);

        //std::cout << img << std::endl;
        torch::Tensor tensor_image = preprocess.process(image_rgb);
        torch::Tensor test = at::max(tensor_image);
        std::cout << "the maximum is: " << test << std::endl;
        //std::cout << tensor_image << std::endl;
        //waitKey(0);
        torch::Device device(torch::kCPU);
        module.to(device);
        priors.to(device);  // move stuff to CPU
        inputs[0] = tensor_image;
        auto start = high_resolution_clock::now();
        auto outputs = module.forward(inputs).toTuple();
        std::vector<PostProcessing::Landmark> result =
            detection.process(outputs->elements()[0].toTensor(),
                              outputs->elements()[1].toTensor(),
                              priors.attr("priors").toTensor());
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Time taken by function: at " << img << " "
                  << duration.count() << " microseconds" << std::endl;
        serialize_results(img, result);
    }
}
