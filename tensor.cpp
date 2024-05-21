#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <functional>
#include <set>  

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    double data;
    double grad;
    std::vector<std::shared_ptr<Tensor>> children;
    std::string op;
    std::function<void()> backward;

    Tensor(double data, std::vector<std::shared_ptr<Tensor>> children = {}, std::string op = "")
        : data(data), grad(0.0), children(children), op(op), backward([]() {}) {}

    void set_backward(std::function<void()> func) {
        backward = func;
    }

   
    std::shared_ptr<Tensor> operator+(Tensor& other) {
        auto out = std::make_shared<Tensor>(this->data + other.data, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other.shared_from_this()}, "+");
        out->set_backward([this, out, &other]() {
            this->grad += out->grad;
            other.grad += out->grad;
            
            // std::cout << "Add Data: " << out->data << ", this grad: " << this->grad << ", other grad: " << other.grad << "\n";

            
        });
        return out;
    }

    std::shared_ptr<Tensor> operator-(Tensor& other) {
        auto out = std::make_shared<Tensor>(this->data - other.data, std::vector<std::shared_ptr<Tensor>>{shared_from_this(),other.shared_from_this()}, "-");
        out->set_backward([this, out, &other]() {
            this->grad += out->grad;
            other.grad -= out->grad;
        });
        return out;
    }

    std::shared_ptr<Tensor> operator*(Tensor& other) {
        
        auto out = std::make_shared<Tensor>(this->data * other.data, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other.shared_from_this()}, "*");
        
        out->set_backward([this, out, &other]() {
            this->grad += out->grad * other.data;
            other.grad += out->grad * this->data;

            // std::cout << "Mul Data: " << out->data << ", this grad: " << this->grad << ", other grad: " << other.grad << "\n";
            
            
        });
        return out;
    }

    std::shared_ptr<Tensor> sigmoid() {
        double exp_val, sig;
        if (this->data >= 0) {
            exp_val = std::exp(-this->data);
            sig = 1 / (1 + exp_val);
        } else {
            exp_val = std::exp(this->data);
            sig = exp_val / (1 + exp_val);
        }
        auto out = std::make_shared<Tensor>(sig, std::vector<std::shared_ptr<Tensor>>{shared_from_this()}, "sigmoid");
        out->set_backward([this, sig, out]() {
            this->grad += (sig * (1 - sig)) * out->grad;
        });
        return out;
    }

    std::shared_ptr<Tensor> log() {
        double epsilon = 1e-10;
        auto out = std::make_shared<Tensor>(std::log(this->data + epsilon), std::vector<std::shared_ptr<Tensor>>{shared_from_this()}, "log");
        out->set_backward([this, out]() {
            this->grad += (1 / (this->data + 1e-8)) * out->grad;
            std::cout<<"Log Grad: "<<this->grad<<"\n";
        });
        return out;
    }

    std::shared_ptr<Tensor> power(double exponent) {
        auto out = std::make_shared<Tensor>(std::pow(this->data, exponent), std::vector<std::shared_ptr<Tensor>>{shared_from_this()}, "pow");
        out->set_backward([this_ptr = shared_from_this(), exponent, out]() {
            this_ptr->grad += (exponent * std::pow(this_ptr->data, exponent - 1)) * out->grad;
        });
        return out;
    }

    void backward_pass() {
        std::vector<std::shared_ptr<Tensor>> all_nodes;
        std::vector<std::shared_ptr<Tensor>> to_visit{shared_from_this()};

        while (!to_visit.empty()) {
            auto node = to_visit.back();
            to_visit.pop_back();
            all_nodes.push_back(node);
            for (auto& child : node->children) {
                to_visit.push_back(child);
            }
        }

        this->grad = 1.0;
        for (auto it = all_nodes.rbegin(); it != all_nodes.rend(); ++it) {
            (*it)->backward();
        }
    }

    void print_graph(int level = 0, std::set<std::shared_ptr<Tensor>>* visited = nullptr) {
        if (!visited) {
            visited = new std::set<std::shared_ptr<Tensor>>();
        }
        
        for (int i = 0; i < level; ++i) {
            std::cout << "  "; 
        }
        std::cout << "Tensor(data=" << data << ", grad=" << grad << ", op=" << op << ")\n";
        
        visited->insert(shared_from_this());

        for (auto& child : children) {
            if (visited->find(child) == visited->end()) { 
                child->print_graph(level + 1, visited);
            }
        }

        if (level == 0) {
            delete visited;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor(data=" << tensor.data << ", grad=" << tensor.grad << ")";
        return os;
    }
};

// int main() {
//     auto a = std::make_shared<Tensor>(5);
//     auto b = std::make_shared<Tensor>(10);
//     auto c = *a**b;
//     auto d = c->power(2);
//     auto e = d->log();
//     // auto b = std::make_shared<Tensor>(3);
//     // auto c = *a + *b;
//     // auto d = c->log();
//     // std::cout << c->data << std::endl;



//     e->backward_pass();    // std::cout << a->grad << std::endl;
//     e->print_graph();
   
//     return 0;
// }