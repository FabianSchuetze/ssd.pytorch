#include <torch/script.h>  // One-stop header.

class Detect {
   public:
    Detect(int num_classes, int bkg_label, int top_k, float conf_thresh,
           float nms_thresh);

    torch::Tensor forward(torch::Tensor localization, torch::Tensor confidence,
                          torch::Tensor priors);
    torch::Tensor decode(const torch::Tensor& localization,
                         const torch::Tensor& priors,
                         const std::vector<float>& variances);

   private:
    int _num_classes, _bkg_label, _top_k, _conf_thresh, _nms_thresh;
};
