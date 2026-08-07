import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kecia.core.tensor5d import Tensor5D

def test_tensor_api():
    print("--- 💎 Testing Tensor5D API & 3-Tier Fallback ---\n")

    # ইউজার জাস্ট সাধারণ লিস্ট বা নামপাই অ্যারে পাস করবে
    inputs = Tensor5D([1.5, -2.0, 3.0])
    weights = Tensor5D([
        [ 2.0, -1.0,  0.5],
        [-3.0,  4.0, -2.0],
        [ 1.5,  0.0,  3.0]
    ])

    print("Inputs Device:", inputs.device)
    print("Weights Device:", weights.device)
    
    print("\n--- 1. Default Run (Auto-selected Photonic) ---")
    # ম্যাজিক: ইউজার শুধু '@' ব্যবহার করবে, ফিজিক্স ব্যাকগ্রাউন্ডে কাজ করবে!
    output_photonic = inputs @ weights
    print(output_photonic)

    print("\n--- 2. Fallback to CPU Test ---")
    # PyTorch-এর মতো সহজেই ডিভাইস সুইচ করা
    inputs_cpu = inputs.to('cpu')
    weights_cpu = weights.to('cpu')
    
    output_cpu = inputs_cpu @ weights_cpu
    print(output_cpu)

    # ভেরিফিকেশন
    print("\n✅ Verification:")
    if str(output_photonic.data) == str(output_cpu.data):
        print("Success! Both hardware tiers produced the exact same result.")

if __name__ == "__main__":
    test_tensor_api()