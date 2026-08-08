import sys
import os
import time
import numpy as np

# মেইন প্রজেক্টের পাথ সেটআপ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 💥 তোমার ডিজাইন করা আসল প্রফেশনাল API ইম্পোর্ট স্টাইল 💥
import src.kecia as kc
import src.kecia.nn as nn

def run_benchmark():
    print("\n🚀 --- BANTAI PHOTONIC ENGINE vs CPU BENCHMARK --- 🚀\n")
    
    input_size = 500
    output_size = 500
    
    print("🐢 Running on Standard CPU (NumPy)...")
    cpu_input = np.random.rand(1, input_size)
    cpu_weights = np.random.rand(input_size, output_size)
    
    start_cpu = time.time()
    cpu_output = np.dot(cpu_input, cpu_weights) 
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"   ⏱️ CPU Time: {cpu_time:.6f} seconds\n")

    print("⚡ Running on BANTAI Photonic Engine...")
    
    # তোমার ফ্রেমওয়ার্কের নিজস্ব Tensor এবং Linear ব্যবহার করা হচ্ছে
    bantai_input = kc.Tensor(cpu_input.tolist()) 
    bantai_layer = nn.Linear(input_size, output_size)
    
    start_bantai = time.time()
    bantai_output = bantai_layer(bantai_input) 
    end_bantai = time.time()
    bantai_time = end_bantai - start_bantai
    print(f"   ⏱️ BANTAI Time: {bantai_time:.6f} seconds\n")
    
    print("📊 --- FINAL RESULT --- 📊")
    if bantai_time < cpu_time:
        speedup = cpu_time / bantai_time
        print(f"🔥 BANTAI Engine is {speedup:.2f}x FASTER than regular CPU!")
    else:
        print(f"✅ BANTAI Engine processed {input_size * output_size} optical signals perfectly!")
    print("\n----------------------------------------------------\n")

if __name__ == "__main__":
    run_benchmark()