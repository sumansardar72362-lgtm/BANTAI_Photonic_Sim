import sys
import os
import time
import numpy as np

# পাইথনকে মেইন প্রজেক্টের ঠিকানা চিনিয়ে দেওয়া হচ্ছে
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kecia.core.tensor5d import Tensor5D
from src.kecia.nn.linear5d import Linear5D

def run_3_layer_network():
    print("\n🌐 --- BANTAI PHOTONIC 3-LAYER NETWORK TEST --- 🌐\n")
    
    # 1. নেটওয়ার্ক ডিজাইন (Architecture)
    print("🛠️ Building Neural Network Architecture...")
    # ইনপুট ১০টি ফিচার, প্রথম হিডেন লেয়ারে ৬৪টি নিউরন
    layer1 = Linear5D(in_features=10, out_features=64)
    # দ্বিতীয় হিডেন লেয়ারে ৩২টি নিউরন
    layer2 = Linear5D(in_features=64, out_features=32)
    # আউটপুট লেয়ারে ২টি নিউরন (ধরে নিচ্ছি, Cat vs Dog ক্লাসিফিকেশন)
    layer3 = Linear5D(in_features=32, out_features=2)
    print("✅ Network Built Successfully!\n")

    # 2. ইনপুট ডেটা তৈরি
    print("📥 Preparing Input Data...")
    # 1x10 সাইজের একটি ডামি ইনপুট তৈরি করা হচ্ছে
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    input_tensor = Tensor5D(dummy_input)
    print(f"Input Shape: {input_tensor.data.shape}\n")

    # 3. ফরোয়ার্ড পাস (Hardware Execution)
    print("🚀 Starting Forward Pass via Photonic Driver...\n")
    start_time = time.time()
    
    # Layer 1
    print("--> Passing through Layer 1 (10 -> 64)")
    out1 = layer1(input_tensor)
    # একটি সিম্পল অ্যাক্টিভেশন ফাংশন (ReLU: নেগেটিভ ভ্যালুগুলোকে ০ করে দেওয়া)
    out1.data = np.maximum(0, out1.data)
    
    # Layer 2
    print("--> Passing through Layer 2 (64 -> 32)")
    out2 = layer2(out1)
    out2.data = np.maximum(0, out2.data) # ReLU
    
    # Layer 3 (Final Output)
    print("--> Passing through Layer 3 (32 -> 2)")
    final_output = layer3(out2)
    
    end_time = time.time()

    # 4. রেজাল্ট প্রিন্ট
    print("\n📊 --- FINAL RESULT --- 📊")
    print(f"Output Tensor Data:\n{final_output.data}")
    print(f"Prediction: {'Dog (Class 1)' if final_output.data[0][1] > final_output.data[0][0] else 'Cat (Class 0)'}")
    print(f"\n⏱️ Total Execution Time: {(end_time - start_time):.6f} seconds")
    print("✅ 3-Layer Photonic Network test completed flawlessly!\n")

if __name__ == "__main__":
    run_3_layer_network()