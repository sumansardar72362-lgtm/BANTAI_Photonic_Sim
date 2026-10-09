import sys
import os
import numpy as np

# src ফোল্ডার কানেক্ট করা হচ্ছে
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D
from kecia.nn import Conv2D, MaxPool2D, Flatten, Dropout, Linear5D

print("🧠 Initializing BANTAI Vision CNN Architecture...\n")

# ১. লেয়ারগুলো সাজানো (Protikriya-এর জন্য মিনি আর্কিটেকচার)
conv1 = Conv2D(in_channels=3, out_channels=8, kernel_size=3, padding=1)  # ভিডিওর প্যাটার্ন বুঝবে
pool1 = MaxPool2D(pool_size=2, stride=2)                                 # সাইজ ছোট করবে
flatten = Flatten()                                                      # চ্যাপ্টা করবে
dropout = Dropout(drop_rate=0.3)                                         # ওভারফিটিং ঠেকাবে
# ইনপুট ডাইমেনশন: 8 চ্যানেল * (16x16 পিক্সেল) = 2048
linear1 = Linear5D(1800, 1)                                              # ফাইনাল ডিসিশন নেবে (Spam/Clean)                                          # ফাইনাল ডিসিশন নেবে (Spam/Clean)                     # ফাইনাল ডিসিশন নেবে (Spam/Clean)

# ২. একটি ডামি ভিডিও ফ্রেম তৈরি করা (Batch=1, Channels=3 (RGB), Height=32, Width=32)
print("🎥 Generating Dummy Video Frame (32x32 RGB pixels)...")
dummy_frame = Tensor5D(np.random.randn(1, 3, 32, 32))

# ৩. ফরোয়ার্ড পাস (মডেলের ভেতর দিয়ে ভিডিও চালানো)
print("\n⏳ Running frame through Photonic Vision Architecture...")

out = conv1.forward(dummy_frame)
print(f"   ➔ After Conv2D: {out.data.shape} (Features Extracted)")

out = pool1.forward(out)
print(f"   ➔ After MaxPool2D: {out.data.shape} (Downsampled)")

out = flatten.forward(out)
print(f"   ➔ After Flatten: {out.data.shape} (Converted to 1D Array)")

out = dropout.forward(out)
print(f"   ➔ After Dropout: {out.data.shape} (Regularized)")

final_prediction = linear1.forward(out)
print(f"   ➔ Final Output: {final_prediction.data.shape} (Prediction Score)")

print(f"\n✅ SUCCESS! Raw Spam Probability Score: {final_prediction.data[0][0]:.4f}")