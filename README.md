<!-- # BANTAI Photonic Compute Stack
# BANTAI_Photonic_Sim -->
# BANTAI Photonic Compute Stack
[![PyPI version](https://badge.fury.io/py/bantai-kecia.svg)](https://pypi.org/project/bantai-kecia/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
**A cutting-edge Deep Learning framework simulating Diamond-Glass Hybrid Photonic Chips.**

BANTAI Kecia is a lightweight, high-performance neural network SDK that simulates mathematical interference of light waves for deep learning computations (Optical MAC Unit) and uses a custom 5D Glass Memory system (`.kecia` format) for zero-data-loss weight storage.

## ✨ Key Features
* **Optical MAC Simulation:** Performs Vector-Matrix Multiplication (VMM) at simulated speed-of-light using light intensity and phase.
* **5D Glass Memory System:** Securely saves and loads trained models using the proprietary `.kecia` format.
* **Hardware Fallback Logic:** 3-Tier automatic execution fallback (Photonic -> CUDA -> CPU).
* **Clean PyTorch-like API:** User-friendly and production-ready interface.
* **Built-in Autograd:** Supports custom Backpropagation and SGD Optimizer.

## 📦 Installation
You can easily install the BANTAI SDK via pip:
```bash
pip install bantai-kecia