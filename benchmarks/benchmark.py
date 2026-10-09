import time
import sys
import os
import random

# src ফোল্ডারটিকে পাথ হিসেবে চিনিয়ে দেওয়া হচ্ছে
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D

def pure_python_matmul(A, B):
    """সাধারণ পাইথন লুপ দিয়ে ম্যাট্রিক্স গুণ (খুবই স্লো)"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0.0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def run_benchmark():
    print("🚀 BANTAI Kecia Photonic Engine - Dynamic Benchmark 🚀")
    print("=" * 60)
    
    # 🎛️ ইউজারের কাছ থেকে ডাইনামিক ইনপুট নেওয়া
    try:
        size_input = input("🎛️ Enter Matrix Size (e.g. 100, 200, 500) [Default is 150]: ")
        MATRIX_SIZE = int(size_input) if size_input.strip() else 150
    except ValueError:
        print("⚠️ Invalid input! Using default size 150.")
        MATRIX_SIZE = 150
        
    print("-" * 60)
    print(f"📊 Generating {MATRIX_SIZE}x{MATRIX_SIZE} random data matrices for the race...")
    
    # র‍্যান্ডম ডেটা জেনারেট করা হচ্ছে
    matrix_a = [[random.uniform(0, 1) for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]
    matrix_b = [[random.uniform(0, 1) for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]
    
    print("⚙️ Loading data into BANTAI Photonic Tensors...")
    tensor_a = Tensor5D(matrix_a)
    tensor_b = Tensor5D(matrix_b)

    print("-" * 60)
    # ---------------------------------------------------------
    print("🐢 1. Testing Standard Python Core (Please wait, this is slow)...")
    start_time = time.time()
    _ = pure_python_matmul(matrix_a, matrix_b)
    python_time = time.time() - start_time
    print(f"   ⏱️ CPU Time: {python_time:.4f} seconds")

    # ---------------------------------------------------------
    print("\n⚡ 2. Testing Photonic C++ Core (BANTAI VMM)...")
    start_time = time.time()
    _ = tensor_a @ tensor_b  # তোমার C++ ইঞ্জিন কল হচ্ছে!
    photonic_time = time.time() - start_time
    
    # জিরো-ডিভিশন এরর ঠেকানোর জন্য
    if photonic_time <= 0: photonic_time = 0.000001 
    print(f"   ⏱️ Photonic Time: {photonic_time:.4f} seconds")

    # ---------------------------------------------------------
    print("=" * 60)
    speedup = python_time / photonic_time
    print(f"🏆 RESULT: BANTAI Photonic Engine is {speedup:.2f}x FASTER than Standard Python!")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()