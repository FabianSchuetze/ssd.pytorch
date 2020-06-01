#ifndef data_processing_hpp
#define data_processing_hpp
#include <torch/script.h>  // One-stop header.
#include <opencv2/core/core.hpp>

class PostProcessing {
   public:
    PostProcessing(std::string);
    torch::Tensor process(const torch::Tensor& localization,
                          const torch::Tensor& confidence,
                          const torch::Tensor& priors);

   private:
    torch::Tensor decode(const torch::Tensor& localization,
                         const torch::Tensor& priors);

    int _num_classes, _bkg_label;
    float _conf_thresh, _nms_thresh;
    std::vector<float> _variances;
};

class PreProcessing {
   public:
    PreProcessing(const std::string&);
    torch::Tensor process(const cv::Mat&);
   private:
    int _size;
};
#endif
