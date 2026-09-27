import sys
import os
import numpy as np

# src ফোল্ডার কানেক্ট করা
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D
from kecia.nn import Conv2D, MaxPool2D, Flatten, Linear5D, ReLU5D, Sequential

def test_photonic_vision():
    print("\n👁️ --- BANTAI Kecia: Photonic Vision Core Test --- 👁️\n")

    # ১. একটি ডামি ভিডিও ফ্রেম তৈরি করা (Batch=1, Channels=3 [RGB], Height=64, Width=64)
    # ধরি এটা তোমার Protikriya প্ল্যাটফর্মের একটি 64x64 পিক্সেলের কালার ফ্রেম
    dummy_frame = np.random.rand(1, 3, 64, 64).astype(np.float32)
    X = Tensor5D(dummy_frame)

    print(f"🎬 Input Video Frame Shape: {X.shape} (Batch, Channels, Height, Width)\n")

    print("🛠️ Building Vision Architecture...")
    
    # যেহেতু আমাদের Conv2D এবং MaxPool2D আপাতত ডামি আউটপুট দিচ্ছে (শেপ কমাচ্ছে না),
    # তাই Flatten হওয়ার পর ডেটার সাইজ হবে 3 * 64 * 64 = 12288
    # আগের লাইনটি: flat_features = 3 * 64 * 64
    # নতুন আপডেট করা লাইন:
    flat_features = 16 * 31 * 31  # (16 channels * 31 height * 31 width)

    # ২. Photonic Vision Model তৈরি করা
    vision_model = Sequential(
        # --- Feature Extraction (চোখ/অপটিক্যাল লেন্স) ---
        Conv2D(in_channels=3, out_channels=16, kernel_size=3),
        ReLU5D(),
        MaxPool2D(kernel_size=2),
        
        # --- Transition ---
        Flatten(),
        
        # --- Decision Making (মস্তিষ্ক) ---
        Linear5D(in_features=flat_features, out_features=128),
        ReLU5D(),
        Linear5D(in_features=128, out_features=1)
    )

    # ৩. ফরোয়ার্ড পাস (ভিডিও ফ্রেম প্রসেস করা)
    print("\n🚀 Processing Video Frame through Photonic Vision Core...\n")
    output = vision_model(X)
    
    print("\n✅ Vision Processing Complete!")
    print(f"📊 Final Model Output (Review Score/Probability): {output.data.flatten()[0]:.4f}")

if __name__ == "__main__":
    test_photonic_vision()