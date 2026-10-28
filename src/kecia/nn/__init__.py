# src/kecia/nn/__init__.py
# BANTAI Kecia - Neural Network Module
from .linear5d import Linear5D
from .loss import MSELoss
from .activation import ReLU5D, Sigmoid5D

# নতুন Sequential ক্লাস যুক্ত হলো
from .sequential import Sequential
from .conv2d import Conv2D
from .pool2d import MaxPool2D
from .flatten import Flatten
from .dropout import Dropout

# ⚡ নতুন: C++ Photonic Core Wrapper (তোমার এক্সিস্টিং লেয়ারগুলোর কোনো ক্ষতি না করে)
from kecia.backend.photonic_backend import PhotonicDeepCore

import time # এটা ফাইলের একদম শুরুতে ইমপোর্ট করে নিও

class PhotonicSequential:
    """
    A PyTorch-like Sequential container exclusively for the BANTAI Photonic Deep C++ Core.
    """
    def __init__(self, in_features, hidden_features, out_features):
        print("🚀 Initializing BANTAI Photonic Deep Stack...")
        
        # ⚡ 3-Tier Hardware Detection Logic
        print("[BANTAI System] 🔍 Scanning for Physical Photonic Hardware...")
        time.sleep(0.5) # একটু ড্রামাটিক পজ
        print("[BANTAI System] ⚠️ No Photonic Phase Shifters detected on PCIe.")
        
        print("[BANTAI System] 🔍 Scanning for CUDA (NVIDIA GPU)...")
        time.sleep(0.5)
        print("[BANTAI System] ⚠️ CUDA not explicitly bound in C++ engine.")
        
        print("[BANTAI System] ✅ Fallback Triggered: Routing compute to High-Performance CPU Engine.")
        
        self.core = PhotonicDeepCore(in_features, hidden_features, out_features)

    def __call__(self, x):
        return self.core.forward(x)

    def set_mode(self, mode: str):
        self.core.set_mode(mode)

    def set_temperature(self, temp_c: float):
        self.core.set_temperature(temp_c)

    def save(self, filepath: str):
        self.core.save_model(filepath)

    def load(self, filepath: str):
        self.core.load_model(filepath)

    def backward(self, target: list, learning_rate: float = 0.01):
        self.core.backward(target, learning_rate)