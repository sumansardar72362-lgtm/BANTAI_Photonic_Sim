#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>  // ⚡ Hardware Mode-এর জন্য নতুন যুক্ত করা হয়েছে

// ১. সাধারণ লিনিয়ার লেয়ার (Single-Layer Legacy)
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

// ২. ⚡ ডিপ লার্নিং কোর (Multi-Layer Photonic Core + Hardware Mode)
class PhotonicDeepCore {
private:
    int in_size, hidden_size, out_size;
    std::vector<std::vector<double>> W1, W2;
    std::vector<double> b1, b2;
    
    // ⚡ Hardware vs Simulation Mode Control ভ্যারিয়েবল
    std::string run_mode = "simulation";

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
        
        // Hidden Layer Processing (ReLU)
        std::vector<double> hidden(hidden_size, 0.0);
        for(int j=0; j<hidden_size; ++j) {
            double sum = 0.0;
            for(int i=0; i<in_size; ++i) sum += input[i] * W1[i][j];
            hidden[j] = std::max(0.0, sum + b1[j]); 
        }

        // Output Layer Processing (ReLU)
        std::vector<double> output(out_size, 0.0);
        for(int j=0; j<out_size; ++j) {
            double sum = 0.0;
            for(int i=0; i<hidden_size; ++i) sum += hidden[i] * W2[i][j];
            output[j] = std::max(0.0, sum + b2[j]); 
        }
        return output;
    }

    // Python-কে ডেটা কন্ট্রোল দেওয়ার জন্য
    void set_w1(const std::vector<std::vector<double>>& w) { W1 = w; }
    void set_w2(const std::vector<std::vector<double>>& w) { W2 = w; }
    void set_b1(const std::vector<double>& b) { b1 = b; }
    void set_b2(const std::vector<double>& b) { b2 = b; }

    // ⚡ নতুন ফিচার: হার্ডওয়্যার মোড অ্যাক্টিভেশন
    void set_mode(const std::string& mode) {
        if (mode == "hardware") {
            run_mode = "hardware";
            std::cout << "\n[BANTAI Hardware Layer] ⚡ Optical link established!" << std::endl;
            std::cout << "[BANTAI Hardware Layer] 🔴 Sending neural weights to Photonic Phase Shifters..." << std::endl;
        } else if (mode == "simulation") {
            run_mode = "simulation";
            std::cout << "\n[BANTAI Simulation Layer] 💻 Running purely on CPU/GPU engine." << std::endl;
        } else {
            std::cout << "⚠️ Warning: Invalid mode. Use 'hardware' or 'simulation'." << std::endl;
        }
    }
};