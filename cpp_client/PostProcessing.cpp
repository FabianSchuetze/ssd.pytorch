#include "DataProcessing.hpp"
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torchvision/nms.h>
#include <torchvision/vision.h>

#include "LoadConfig.hpp"

using torch::Tensor;

PostProcessing::PostProcessing(std::string config)
    : _num_classes(0),
      _bkg_label(0),
      _conf_thresh(0),
      _nms_thresh(0),
      _variances(2) {
    std::ifstream paramFile{"params.txt"};
    std::map<std::string, std::string> params{
        std::istream_iterator<kv_pair>{paramFile},
        std::istream_iterator<kv_pair>{}};
    _num_classes = std::stoi(params["num_classes"]);
    _bkg_label = std::stoi(params["bkg_label"]);
    _conf_thresh = std::stof(params["conf_thresh"]);
    _nms_thresh = std::stof(params["nms_thresh"]);
    _variances[0] = std::stof(params["variance_0"]);
    _variances[1] = std::stof(params["variance_1"]);

    ;
}

Tensor PostProcessing::decode(const Tensor& loc, const Tensor& priors) {
    Tensor left = priors.slice(1, 0, 2) +
                  loc.slice(1, 0, 2) * _variances[0] * priors.slice(1, 2);
    Tensor right =
        priors.slice(1, 2) + torch::exp(loc.slice(1, 0, 2) * _variances[1]);
    Tensor boxes = torch::cat({left, right}, 1);
    return boxes;
}

Tensor PostProcessing::process(const Tensor& localization, const Tensor& confidence,
                               const Tensor& priors) {
    int num_priors = priors.size(0);
    Tensor output = torch::empty({0, 5});
    Tensor conf_preds = confidence.view({num_priors, _num_classes});
    conf_preds = conf_preds.transpose(1, 0);
    Tensor boundig_boxes = localization.squeeze(0);
    Tensor decoded_boxes = decode(localization, priors);
    Tensor conf_scores = conf_preds.clone();
    for (int i = 1; i < _num_classes; ++i) {
        Tensor cur = conf_scores.slice(0, i, i + 1);
        Tensor confident = cur.gt(_conf_thresh);
        Tensor scores = cur.masked_select(confident);
        Tensor l_mask = confident.transpose(1, 0).expand_as(decoded_boxes);
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
