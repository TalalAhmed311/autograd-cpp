#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include "tensor.cpp" 

class Neuron {
public:
    std::vector<std::shared_ptr<Tensor>> w;
    std::shared_ptr<Tensor> b;

    Neuron(int nin) {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < nin; ++i) {
            w.push_back(std::make_shared<Tensor>(dis(gen)));
        }
        b = std::make_shared<Tensor>(dis(gen));
    }

    std::shared_ptr<Tensor> operator()(const std::vector<std::shared_ptr<Tensor>>& x) {
        std::shared_ptr<Tensor> out = *w[0] * *x[0]; 
        for (int i = 1; i < w.size(); ++i) {
            out = *out + *(*w[i] * *x[i]);
        }
        out = *out + *b;
        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> params = w;
        params.push_back(b);
        return params;
    }
};

// int main() {
//     Neuron n(5);

//     std::vector<std::shared_ptr<Tensor>> inputs;
//     for (int i = 0; i < 5; ++i) {
//         inputs.push_back(std::make_shared<Tensor>(118.2)); // Example input of 1.0 for each input
//     }

//     auto output = n(inputs);
//     auto out = output->log();
//     std::cout << "Neuron output: " << out->data << std::endl;

//     out->backward_pass();
//     // out->backward_pass();
//     // out->backward_pass();
//     // out->backward_pass();
//     // out->backward_pass();
    
//     // std::cout << "Gradient of the bias: " << n.w[3]->grad << std::endl;
//     out->print_graph();
//     return 0;
// }
