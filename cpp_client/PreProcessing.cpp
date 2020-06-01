#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataProcessing.hpp"
#include "LoadConfig.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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
    at::Tensor tensor_img =
        torch::from_blob(transformed.data, {1, 3, _size, _size});
    tensor_img = tensor_img.to(at::kFloat);
    return tensor_img;
}
