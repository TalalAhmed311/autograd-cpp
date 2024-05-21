#include "layer.cpp"
#include <iostream>
#include <vector>
#include <memory>
#include <string>

class MLP {
public:
    int in_feat;
    std::vector<int> out_features;
    std::vector<std::shared_ptr<Layer>> layers;

    MLP(int in_feat, std::vector<int> out_features) : in_feat(in_feat), out_features(out_features) {
        std::vector<std::string> layer_names;
        for (size_t i = 0; i < out_features.size(); ++i) {
            layer_names.push_back("layer_" + std::to_string(i));
        }

        std::vector<int> size = {in_feat};
        size.insert(size.end(), out_features.begin(), out_features.end());

        for (size_t i = 0; i < out_features.size(); ++i) {
            layers.push_back(std::make_shared<Layer>(size[i], size[i + 1]));
        }
    }

    std::vector<std::shared_ptr<Tensor>> operator()(const std::vector<std::shared_ptr<Tensor>>& x) {
        std::vector<std::shared_ptr<Tensor>> output = x;
        for (auto& layer : layers) {
            output = (*layer)(output);
        }
        return output;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> params;
        for (auto& layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    void zero_grad() {
        for (auto& p : parameters()) {
            p->grad = 0;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const MLP& mlp) {
        os << "MLP(in_feat=" << mlp.in_feat << ", out_features={";
        for (size_t i = 0; i < mlp.out_features.size(); ++i) {
            os << mlp.out_features[i];
            if (i != mlp.out_features.size() - 1) {
                os << ", ";
            }
        }
        os << "})";
        return os;
    }
};

int main() {
    int in_features = 3;
    std::vector<int> out_features = {3, 2, 1};
    MLP mlp(in_features, out_features);

    std::vector<std::shared_ptr<Tensor>> input;
    for (int i = 0; i < in_features; ++i) {
        input.push_back(std::make_shared<Tensor>(1.0)); 
    }

    std::vector<std::shared_ptr<Tensor>> output = mlp(input);
    output[0]->backward_pass();
  

    std::cout << "Output: " << std::endl;
    for (auto& out : output) {
        out->print_graph(); // Assuming Tensor class has a print method
    }



    std::cout << "Gradients before zero_grad: " << std::endl;
    for (auto& param : mlp.parameters()) {
        std::cout << *param.get() << std::endl;
    }

    mlp.zero_grad();

    std::cout << "Gradients after zero_grad: " << std::endl;
    for (auto& param : mlp.parameters()) {
        std::cout << *param.get() << std::endl;
    }
    
    

    return 0;
}