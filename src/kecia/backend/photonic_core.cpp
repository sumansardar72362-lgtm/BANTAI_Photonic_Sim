#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "photonic_layer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(photonic_backend, m) {
    m.doc() = "BANTAI Kecia Photonic Neural Engine Backend";

    py::class_<PhotonicLinear>(m, "PhotonicLinear")
        .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"))
        .def("forward", &PhotonicLinear::forward)
        .def("get_weights", &PhotonicLinear::get_weights)
        .def("set_weights", &PhotonicLinear::set_weights)
        .def("set_biases", &PhotonicLinear::set_biases);

    py::class_<PhotonicDeepCore>(m, "PhotonicDeepCore")
        .def(py::init<int, int, int>(), py::arg("in_features"), py::arg("hid_features"), py::arg("out_features"))
        .def("forward", &PhotonicDeepCore::forward)
        .def("set_w1", &PhotonicDeepCore::set_w1)
        .def("set_w2", &PhotonicDeepCore::set_w2)
        .def("set_b1", &PhotonicDeepCore::set_b1)
        .def("set_b2", &PhotonicDeepCore::set_b2)
        .def("set_mode", &PhotonicDeepCore::set_mode)
        .def("save_model", &PhotonicDeepCore::save_model)
        .def("load_model", &PhotonicDeepCore::load_model)
        .def("set_temperature", &PhotonicDeepCore::set_temperature) // আগের লাইনের সেমিকোলন মুছে দাও
        .def("backward", &PhotonicDeepCore::backward); // 👈 নতুন শেষ লাইন
}