import sys
import os

# src ফোল্ডারটিকে পাথ হিসেবে চিনিয়ে দেওয়া হচ্ছে
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D

print("🔥 Testing Tensor5D with C++ Photonic Backend!\n")

# দুটি 2D ডামি ম্যাট্রিক্স তৈরি করা হচ্ছে
matrix_a = [
    [1.5, 2.0, 3.0],
    [0.5, 1.0, 1.5]
]

matrix_b = [
    [0.5, 1.0],
    [1.5, 2.0],
    [2.0, 3.0]
]

# সেগুলোকে তোমার নতুন Tensor5D অবজেক্টে কনভার্ট করা হলো
tensor_a = Tensor5D(matrix_a)
tensor_b = Tensor5D(matrix_b)

print("Matrix A:")
print(f"{tensor_a}\n")
print("Matrix B:")
print(f"{tensor_b}\n")

print("-" * 50)

# ম্যাট্রিক্স গুণন (Photonic VMM via C++)
# এখানেই আসল ম্যাজিকটা ঘটবে!
result = tensor_a @ tensor_b

print("\n🚀 Final Result from C++ Engine:")
print(result)