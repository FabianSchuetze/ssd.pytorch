#include <torch/script.h>  // One-stop header.

#include <chrono>
#include <iostream>
#include <memory>

#include "DataProcessing.hpp"

using namespace std::chrono;
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
    PostProcessing detection("params.txt");
    for (int i = 0; i < 10; ++i) {
        inputs[0] = torch::rand({1, 3, 300, 300});
        auto start = high_resolution_clock::now();
        auto outputs = module.forward(inputs).toTuple();
        auto final = detection.process(outputs->elements()[0].toTensor(),
                                       outputs->elements()[1].toTensor(),
                                       priors.attr("priors").toTensor());
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Time taken by function: at " << i << " "
                  << duration.count() << " microseconds" << std::endl;
    }
}
