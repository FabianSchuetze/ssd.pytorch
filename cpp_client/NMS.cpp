#include "NMS.hpp"

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torchvision/nms.h>
#include <torchvision/vision.h>

using torch::Tensor;

Detect::Detect(int num_classes, int bkg_label, int top_k, float conf_thresh,
               float nms_thresh)
    : _num_classes(num_classes),
      _bkg_label(bkg_label),
      _top_k(top_k),
      _conf_thresh(conf_thresh),
      _nms_thresh(nms_thresh) {
    ;
}

Tensor Detect::decode(const Tensor& loc, const Tensor& priors,
                      const std::vector<float>& variances) {
    Tensor left = priors.slice(1, 0, 2) +
                  loc.slice(1, 0, 2) * variances[0] * priors.slice(1, 2);
    Tensor right =
        priors.slice(1, 2) + torch::exp(loc.slice(1, 0, 2) * variances[1]);
    Tensor boxes = torch::cat({left, right}, 1);
    return boxes;
}

Tensor Detect::forward(Tensor localization, Tensor confidence, Tensor priors) {
    std::vector<float> variance({0.1, 0.2});
    int num_priors = priors.size(0);
    Tensor output = torch::empty({0, 5});
    Tensor conf_preds = confidence.view({num_priors, _num_classes});
    localization = localization.squeeze(0);
    conf_preds = conf_preds.transpose(1, 0);
    Tensor decoded_boxes = decode(localization, priors, variance);
    Tensor conf_scores = conf_preds.clone();
    for (int i = 1; i < _num_classes; ++i) {
        Tensor cur = conf_scores.slice(0, i, i + 1);
        Tensor c_mask = cur.gt(_conf_thresh);
        Tensor scores = cur.masked_select(c_mask);
        Tensor l_mask = c_mask.transpose(1, 0).expand_as(decoded_boxes);
        Tensor boxes = decoded_boxes.masked_select(l_mask).view({-1, 4});
        Tensor ids = nms_cpu(boxes, scores, _nms_thresh);
        Tensor selected_scores = scores.index_select(0, ids).unsqueeze(1);
        Tensor selected_boxes = boxes.index_select(0, ids);
        Tensor test = torch::cat({selected_scores, selected_boxes}, 1);
        output = torch::cat({output, test}, 0);
    }
    std::cout << "The output size is:" << output.size(0) << std::endl;
    return output;
}
