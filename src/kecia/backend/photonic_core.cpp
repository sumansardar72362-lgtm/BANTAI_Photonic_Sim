#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <complex>

namespace py = pybind11;

// ফোটোনিক VMM (Intensity & Phase Simulation)
std::vector<double> photonic_vmm(const std::vector<double>& amplitudes, const std::vector<double>& phases) {
    std::vector<double> result;
    
    // আলোর তরঙ্গের সুপারপজিশন (Interference) হিসাব করার জন্য Complex Number
    std::complex<double> total_optical_field(0.0, 0.0);
    
    for (size_t i = 0; i < amplitudes.size() && i < phases.size(); ++i) {
        // ফিজিক্স ইকুয়েশন: E = A * e^(i * φ) -> A * (cos(φ) + i * sin(φ))
        std::complex<double> wave(amplitudes[i] * cos(phases[i]), amplitudes[i] * sin(phases[i]));
        total_optical_field += wave;
    }
    
    // ফোটোডিটেক্টর (Photodetector) আলোর ইনটেনসিটি মাপে: I = |E|^2
    double output_intensity = std::norm(total_optical_field); 
    
    result.push_back(output_intensity);
    return result;
}

PYBIND11_MODULE(photonic_backend, m) {
    m.def("photonic_vmm", &photonic_vmm, "Simulate photonic VMM with light amplitude and phase");
}