#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <complex>

namespace py = pybind11;

// ১. পুরোনো Photonic VMM (Linear5D এর জন্য)
std::vector<double> photonic_vmm(const std::vector<double>& amplitudes, const std::vector<double>& phases) {
    std::vector<double> result;
    std::complex<double> total_optical_field(0.0, 0.0);
    for (size_t i = 0; i < amplitudes.size() && i < phases.size(); ++i) {
        std::complex<double> wave(amplitudes[i] * cos(phases[i]), amplitudes[i] * sin(phases[i]));
        total_optical_field += wave;
    }
    result.push_back(std::norm(total_optical_field)); 
    return result;
}

// ২. নতুন Photonic Conv2D (ভিডিও ফ্রেম প্রসেসিংয়ের জন্য)
// অপটিক্যাল ফিল্টার বা লেন্স ইমেজের ওপর দিয়ে স্লাইড করবে
std::vector<double> photonic_conv2d(const std::vector<double>& image, int img_size, const std::vector<double>& filter, int kernel_size) {
    int out_size = img_size - kernel_size + 1;
    std::vector<double> output(out_size * out_size, 0.0);

    for (int y = 0; y < out_size; ++y) {
        for (int x = 0; x < out_size; ++x) {
            double optical_sum = 0.0;
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int img_idx = (y + ky) * img_size + (x + kx);
                    int filter_idx = ky * kernel_size + kx;
                    // ফোটোনিক ইন্টারফারেন্স: পিক্সেল এবং ফিল্টারের সুপারপজিশন
                    optical_sum += image[img_idx] * filter[filter_idx];
                }
            }
            output[y * out_size + x] = optical_sum;
        }
    }
    return output;
}

PYBIND11_MODULE(photonic_backend, m) {
    m.def("photonic_vmm", &photonic_vmm, "Simulate photonic VMM");
    m.def("photonic_conv2d", &photonic_conv2d, "Simulate optical 2D convolution for video frames");
}