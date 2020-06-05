#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torchvision/nms.h>
#include <torchvision/vision.h>

#include <fstream>

#include "DataProcessing.hpp"
#include "LoadConfig.hpp"

using torch::Tensor;
typedef std::vector<PostProcessing::Landmark> landmarks;

PostProcessing::PostProcessing(const std::string& config)
    : _num_classes(0),
      _bkg_label(0),
      _conf_thresh(0),
      _nms_thresh(0),
      _variances(2) {
     std::cout << "the path is for PostProcessing: " << config << std::endl;
    std::ifstream paramFile{config};
    std::map<std::string, std::string> params{
        std::istream_iterator<kv_pair>{paramFile},
        std::istream_iterator<kv_pair>{}};
    std::cout << "the size of the map is: " << params.size() << std::endl;
    for (auto ele : params) {
        std::cout << ele.first << ", " << ele.second << std::endl;
    }
    _num_classes = std::stoi(params["num_classes"]);
    _bkg_label = std::stoi(params["bkg_label"]);
    _conf_thresh = std::stof(params["conf_thresh"]);
    _nms_thresh = std::stof(params["nms_thresh"]);
    _variances[0] = std::stof(params["variance_0"]);
    _variances[1] = std::stof(params["variance_1"]);
    print_arguments();

    ;
}

void PostProcessing::print_arguments() {
    std::cout << "The parameters for the algortihm are: "
              << "num_classes: " << _num_classes << std::endl
              << "bgk_label: " << _bkg_label << std::endl
              << "conf_thresh: " << _conf_thresh << std::endl
              << "nms_thresh: " << _nms_thresh << std::endl
              << " bgk_label: " << _bkg_label << std::endl
              << " variance: " << _variances[0] << ", " << _variances[1]
              << std::endl;
}

//TODO: This is not correct. I need to looks at the python implementation
Tensor PostProcessing::decode(const Tensor& loc, const Tensor& priors) {
    Tensor left = priors.slice(1, 0, 2) +
                  loc.slice(1, 0, 2) * _variances[0] * priors.slice(1, 2);
    Tensor right =
        priors.slice(1, 2) + torch::exp(loc.slice(1, 0, 2) * _variances[1]);
    Tensor boxes = torch::cat({left, right}, 1);
    boxes.slice(1, 0, 2) -=boxes.slice(1, 2) / 2;
    boxes.slice(1, 2) += boxes.slice(1, 0, 2);
    return boxes;
}

landmarks PostProcessing::process(const Tensor& localization,
                                  const Tensor& confidence,
                                  const Tensor& priors,
                                  const std::pair<float, float>& img_size) {
    std::vector<PostProcessing::Landmark> results;
    int num_priors = priors.size(0);
    Tensor output = torch::empty({0, 5});
    Tensor conf_preds = confidence.view({num_priors, _num_classes});
    conf_preds = conf_preds.transpose(1, 0);
    Tensor bounding_boxes = localization.squeeze(0);
    Tensor decoded_boxes = decode(bounding_boxes, priors);
    Tensor conf_scores = conf_preds.clone();
    for (int i = 1; i < _num_classes; ++i) {
        Tensor cur = conf_scores.slice(0, i, i + 1);
        Tensor confident = cur.gt(_conf_thresh);
        Tensor scores = cur.masked_select(confident);
        if (scores.size(0) == 0) {
            continue;
        }
        Tensor l_mask = confident.transpose(1, 0).expand_as(decoded_boxes);
        Tensor boxes = decoded_boxes.masked_select(l_mask).view({-1, 4});
        Tensor ids = nms_cpu(boxes, scores, _nms_thresh);
        Tensor selected_scores = scores.index_select(0, ids);
        Tensor selected_boxes = boxes.index_select(0, ids);
        convert(i, selected_scores, selected_boxes, img_size, results);
        // results.insert(results.end(), tmp.begin(), tmp.end());
    }
    return results;
}

void PostProcessing::convert(int i, const Tensor& scores, const Tensor& boxes,
                             const std::pair<float, float>& img_size,
                             landmarks& results) {
    // std::vector<PostProcessing::Landmark> results;
    //float img_width = img_size.first;
    //float img_height = img_size.second;
    for (int i = 0; i < scores.size(0); ++i) {
        float xmin = boxes[i][0].item<float>() * 300;
        float ymin = boxes[i][1].item<float>() * 300;
        float xmax = boxes[i][2].item<float>() * 300;
        float ymax = boxes[i][3].item<float>() * 300;
        PostProcessing::Landmark l;
        l.xmin = xmin;
        l.ymin = ymin;
        l.xmax = xmax;
        l.ymax = ymax;
        l.confidence = scores[i].item<float>();
        l.label = i;
        results.push_back(l);
    }
}
