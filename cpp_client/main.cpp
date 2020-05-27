#include <torch/script.h>  // One-stop header.
#include "NMS.hpp"

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;
int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    torch::jit::script::Module priors;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
        priors = torch::jit::load(argv[2]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 300, 300}));
    Detect detection(5, 0, 5, 0.01, 0.45);
    auto outputs = module.forward(inputs).toTuple();
    detection.forward(outputs->elements()[0].toTensor(),
                      outputs->elements()[1].toTensor(),
                      priors.attr("priors").toTensor());


    // Execute the model and turn its output into a tensor.
    for (int i = 0; i < 10; ++i) {
        auto start = high_resolution_clock::now();
        auto outputs = module.forward(inputs).toTuple();
        torch::Tensor output = outputs->elements()[0].toTensor();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
         std::cout << "Time taken by function: at "  << i << " " 
             << duration.count() << " microseconds" << std::endl;
    }
    //std::cout << output.slice([>dim=*/1, /*start=*/0, /*end=<]5) << '\n';
}
