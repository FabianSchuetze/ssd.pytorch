#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataProcessing.hpp"
#include "LoadConfig.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
PreProcessing::PreProcessing(const std::string& config) : _size(0) {
    std::cout << "the path is: " << config << std::endl;
    std::ifstream paramFile{config};
    std::map<std::string, std::string> params{
        std::istream_iterator<kv_pair>{paramFile},
        std::istream_iterator<kv_pair>{}};
    //std::string s = params["size"];
    _size = std::stoi(params["size"]);
}

torch::Tensor PreProcessing::process(const cv::Mat& img) {
    cv::Mat transformed;
    cv::Size sz(_size, _size);
    cv::resize(img, transformed, sz);
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", transformed);                   // Show our image inside it.
    //waitKey(0);
    at::Tensor tensor_img =
        torch::from_blob(transformed.data, {1, _size, _size, 3}, at::kByte);
    tensor_img = tensor_img.to(at::kFloat);
    tensor_img = tensor_img.permute({0, 3, 1, 2});
    tensor_img = tensor_img / 255;
    std::cout << "The size of the tensor is " <<
        tensor_img.size(0) << ", " << tensor_img.size(1) << ", " <<
        tensor_img.size(2) << ", " << tensor_img.size(3) << std::endl;
    return tensor_img;
}
