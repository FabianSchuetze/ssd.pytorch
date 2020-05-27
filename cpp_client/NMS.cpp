#include "NMS.hpp"
#include <torch/csrc/autograd/generated/variable_factories.h>

using torch::Tensor;

Detect::Detect(int num_classes, int bkg_label, int top_k, float conf_thresh,
               float nms_thresh)
    : _num_classes(num_classes),
      _bkg_label(bkg_label),
      _top_k(top_k),
      _conf_thresh(conf_thresh),
      _nms_thresh(nms_thresh) {;}

Tensor Detect::decode(const Tensor& localization, const Tensor& priors, 
                      const std::vector<float>& variances) {
    Tensor boxes = torch::cat((
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] + torch.exp([:, :2] * variances[1])), 1);
    return boxes
}
Tensor Detect::forward(Tensor localization, Tensor confidence, Tensor  priors) {
    int num = localization.size(0);
    int num_priors = priors.size(0);
    Tensor output = torch::zeros({num, _num_classes, _top_k, 5});
    Tensor conf_preds = confidence.view({num, num_priors, _num_classes});
    conf_preds = conf_preds.transpose(2,1);
    //Tensor decoded_boxes = decode();

    return torch::ones({300, 300});
}
