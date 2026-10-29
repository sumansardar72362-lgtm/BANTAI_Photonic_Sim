#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <fstream> 

// ১. সাধারণ লিনিয়ার লেয়ার
class PhotonicLinear {
private:
    int input_size;
    int output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

public:
    PhotonicLinear(int in_features, int out_features) {
        input_size = in_features; output_size = out_features;
        std::random_device rd; std::mt19937 gen(rd());
        std::normal_distribution<double> d(0.0, sqrt(2.0 / input_size));
        weights.resize(input_size, std::vector<double>(output_size));
        biases.resize(output_size, 0.0);
        for (int i = 0; i < input_size; ++i)
            for (int j = 0; j < output_size; ++j) weights[i][j] = d(gen);
    }

    std::vector<double> forward(const std::vector<double>& input_vector) {
        if (input_vector.size() != input_size) throw std::invalid_argument("Input size mismatch!");
        std::vector<double> output(output_size, 0.0);
        for (int j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (int i = 0; i < input_size; ++i) sum += input_vector[i] * weights[i][j];
            output[j] = std::max(0.0, sum + biases[j]); 
        }
        return output;
    }
    
    std::vector<std::vector<double>> get_weights() { return weights; }
    void set_weights(const std::vector<std::vector<double>>& new_weights) { weights = new_weights; }
    void set_biases(const std::vector<double>& new_biases) { biases = new_biases; }
};

// ২. ⚡ ডিপ লার্নিং কোর (Smart Autograd + Leaky ReLU)
class PhotonicDeepCore {
private:
    int in_size, hidden_size, out_size;
    std::vector<std::vector<double>> W1, W2;
    std::vector<double> b1, b2;
    std::string run_mode = "simulation";

    double temperature = 25.0; 
    double thermal_noise_factor = 0.0;

    // 🧠 Autograd Memory Cache
    std::vector<double> last_input;
    std::vector<double> last_hidden;
    std::vector<double> last_output;

public:
    PhotonicDeepCore(int in_features, int hid_features, int out_features) {
        in_size = in_features; hidden_size = hid_features; out_size = out_features;
        
        std::random_device rd; std::mt19937 gen(rd());
        std::normal_distribution<double> d1(0.0, sqrt(2.0 / in_size));
        std::normal_distribution<double> d2(0.0, sqrt(2.0 / hidden_size));

        W1.assign(in_size, std::vector<double>(hidden_size, 0.0));
        b1.assign(hidden_size, 0.0);
        for(int i=0; i<in_size; ++i) for(int j=0; j<hidden_size; ++j) W1[i][j] = d1(gen);

        W2.assign(hidden_size, std::vector<double>(out_size, 0.0));
        b2.assign(out_size, 0.0);
        for(int i=0; i<hidden_size; ++i) for(int j=0; j<out_size; ++j) W2[i][j] = d2(gen);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        if (input.size() != in_size) throw std::invalid_argument("Input mismatch!");
        last_input = input; 

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> noise(0.0, thermal_noise_factor);

        // Hidden Layer: Leaky ReLU (Prevents dying neurons)
        last_hidden.assign(hidden_size, 0.0);
        for(int j=0; j<hidden_size; ++j) {
            double sum = 0.0;
            for(int i=0; i<in_size; ++i) {
                double active_weight = W1[i][j] + (run_mode == "simulation" ? noise(gen) : 0.0);
                sum += input[i] * active_weight;
            }
            double z = sum + b1[j];
            last_hidden[j] = (z > 0) ? z : 0.01 * z; // 🧠 Leaky ReLU Injection
        }

        // Output Layer: Linear (Allows unrestricted learning)
        last_output.assign(out_size, 0.0);
        for(int j=0; j<out_size; ++j) {
            double sum = 0.0;
            for(int i=0; i<hidden_size; ++i) {
                double active_weight = W2[i][j] + (run_mode == "simulation" ? noise(gen) : 0.0);
                sum += last_hidden[i] * active_weight;
            }
            last_output[j] = sum + b2[j]; // 🧠 Linear Activation
        }
        return last_output;
    }

    void backward(const std::vector<double>& target, double learning_rate) {
        if (target.size() != out_size) throw std::invalid_argument("Target mismatch!");

        // 1. Output Gradients (Linear derivative is 1, so simply error)
        std::vector<double> dZ2(out_size, 0.0);
        for(int j=0; j<out_size; ++j) {
            dZ2[j] = last_output[j] - target[j]; 
        }

        // 2. Hidden Layer Gradients (Leaky ReLU derivative)
        std::vector<double> dZ1(hidden_size, 0.0);
        for(int i=0; i<hidden_size; ++i) {
            double sum_err = 0.0;
            for(int j=0; j<out_size; ++j) sum_err += dZ2[j] * W2[i][j];
            dZ1[i] = (last_hidden[i] > 0) ? sum_err : 0.01 * sum_err; 
        }

        // 3. Update W2, b2
        for(int i=0; i<hidden_size; ++i) {
            for(int j=0; j<out_size; ++j) W2[i][j] -= learning_rate * (last_hidden[i] * dZ2[j]);
        }
        for(int j=0; j<out_size; ++j) b2[j] -= learning_rate * dZ2[j];

        // 4. Update W1, b1
        for(int i=0; i<in_size; ++i) {
            for(int j=0; j<hidden_size; ++j) W1[i][j] -= learning_rate * (last_input[i] * dZ1[j]);
        }
        for(int j=0; j<hidden_size; ++j) b1[j] -= learning_rate * dZ1[j];
    }

    void set_w1(const std::vector<std::vector<double>>& w) { W1 = w; }
    void set_w2(const std::vector<std::vector<double>>& w) { W2 = w; }
    void set_b1(const std::vector<double>& b) { b1 = b; }
    void set_b2(const std::vector<double>& b) { b2 = b; }

    void set_mode(const std::string& mode) {
        if (mode == "hardware") {
            run_mode = "hardware";
            std::cout << "\n[BANTAI Hardware Layer] ⚡ Optical link established!" << std::endl;
            std::cout << "[BANTAI Hardware Layer] 🔴 Sending neural weights to Photonic Phase Shifters..." << std::endl;
        } else if (mode == "simulation") {
            run_mode = "simulation";
            std::cout << "\n[BANTAI Simulation Layer] 💻 Running purely on CPU/GPU engine." << std::endl;
        } else {
            std::cout << "⚠️ Warning: Invalid mode." << std::endl;
        }
    }

    void set_temperature(double temp_celsius) {
        temperature = temp_celsius;
        if (temperature > 25.0) {
            thermal_noise_factor = (temperature - 25.0) * 0.0005;
        } else {
            thermal_noise_factor = 0.0;
        }
        std::cout << "[Physics Engine] 🌡️ Chip Temperature set to " << temperature 
                  << "°C. Thermal Noise level: " << thermal_noise_factor * 100 << "%" << std::endl;
    }

    void save_model(const std::string& filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if(!out) throw std::runtime_error("Cannot create .kecia file!");
        std::string magic = "KECIA5D";
        out.write(magic.c_str(), magic.size());
        out.write(reinterpret_cast<char*>(&in_size), sizeof(int));
        out.write(reinterpret_cast<char*>(&hidden_size), sizeof(int));
        out.write(reinterpret_cast<char*>(&out_size), sizeof(int));

        for(int i=0; i<in_size; ++i) out.write(reinterpret_cast<const char*>(W1[i].data()), hidden_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(b1.data()), hidden_size * sizeof(double));
        for(int i=0; i<hidden_size; ++i) out.write(reinterpret_cast<const char*>(W2[i].data()), out_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(b2.data()), out_size * sizeof(double));
        
        out.close();
        std::cout << "[5D Glass Memory] 💾 Neural pathways securely crystallized to " << filepath << std::endl;
    }

    void load_model(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if(!in) throw std::runtime_error("Cannot open .kecia file! File missing.");
        char magic[8] = {0};
        in.read(magic, 7);
        if(std::string(magic) != "KECIA5D") throw std::runtime_error("Security Alert: Invalid .kecia format! Glass memory corrupted.");
        int i_s, h_s, o_s;
        in.read(reinterpret_cast<char*>(&i_s), sizeof(int));
        in.read(reinterpret_cast<char*>(&h_s), sizeof(int));
        in.read(reinterpret_cast<char*>(&o_s), sizeof(int));
        if (i_s != in_size || h_s != hidden_size || o_s != out_size) throw std::runtime_error("Architecture Mismatch!");

        for(int i=0; i<in_size; ++i) in.read(reinterpret_cast<char*>(W1[i].data()), hidden_size * sizeof(double));
        in.read(reinterpret_cast<char*>(b1.data()), hidden_size * sizeof(double));
        for(int i=0; i<hidden_size; ++i) in.read(reinterpret_cast<char*>(W2[i].data()), out_size * sizeof(double));
        in.read(reinterpret_cast<char*>(b2.data()), out_size * sizeof(double));
        in.close();
        std::cout << "[5D Glass Memory] 🔮 Photonic weights successfully restored from " << filepath << std::endl;
    }
};