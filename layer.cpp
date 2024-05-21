#include "neuron.cpp"
#include <iostream>
#include <vector>
#include <memory>

class Layer {
public:
    std::vector<std::shared_ptr<Neuron>> neurons;
    int in_feat;
    int out_feat;

    Layer(int in_feat, int out_feat) : in_feat(in_feat), out_feat(out_feat) {
        for (int i = 0; i < out_feat; ++i) {
            neurons.push_back(std::make_shared<Neuron>(in_feat));
        }
    }

    std::vector<std::shared_ptr<Tensor>> operator()(const std::vector<std::shared_ptr<Tensor>>& x) {
        std::vector<std::shared_ptr<Tensor>> output;
        for (int i = 0; i < out_feat; ++i) {
            output.push_back(neurons[i]->operator()(x));
        }
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> params;
        for (auto& neuron : neurons) {
            auto neuron_params = neuron->parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};


// int main() {
//     int in_features = 3;
//     int out_features = 1;
//     Layer layer(in_features, out_features);

//     std::vector<std::shared_ptr<Tensor>> input;
//     for (int i = 0; i < in_features; ++i) {
//         input.push_back(std::make_shared<Tensor>(1.0)); // Example input, replace with actual data
//     }

//     std::vector<std::shared_ptr<Tensor>> output = layer(input);
// //    std::cout << "Output: " << std::endl;
// //     for (auto& out : output) {
// //         out->print_graph(); 
// //     };

//     for(auto& p: layer.parameters()){
//         std::cout << "Param: " << *(p.get()) << std::endl;
//     }
//     return 0;
// }