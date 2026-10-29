# BANTAI Kecia SDK (v2.0.0)
**A Next-Generation Deep Learning Framework Simulating Diamond-Glass Hybrid Photonic Chips.**

[![PyPI version](https://badge.fury.io/py/bantai-kecia.svg)](https://badge.fury.io/py/bantai-kecia)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BANTAI Kecia** is a highly optimized, C++ backed Python SDK designed to simulate the physics and computational mechanics of Photonic Neural Networks. Built for the future of AI hardware, it introduces a 3-tier execution architecture, custom physical thermal noise modeling, and dynamic 5D Glass Memory states.

---

## 🌟 Core Features

*   **⚡ C++ Photonic Core:** Heavy matrix computations are routed through a high-performance C++ backend (`pybind11`), bypassing Python's GIL.
*   **🧠 Built-in Autograd Engine:** Complete forward and backward pass implementation featuring Custom SGD, Leaky ReLU, and dying-neuron prevention.
*   **🔥 Thermal Simulation:** Real-world physics injection simulating hardware temperature fluctuations and phase-shifter precision errors.
*   **💾 5D Glass Memory:** Advanced multi-dimensional state tracking for temporal and spatial data storage.
*   **🔄 3-Tier Hardware Fallback:** Automatic environment scanning and execution routing (`Photonic Hardware` ➔ `CUDA` ➔ `High-Performance CPU`).

---

## 📦 Installation

BANTAI Kecia is available on PyPI. Install it instantly using pip:

```bash
pip install bantai-kecia