import sys
import os
import numpy as np

# মূল ফোল্ডার থেকে মডিউল ইম্পোর্ট করার জন্য পাথ অ্যাড করা হচ্ছে
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kecia.core.glass_memory import GlassMemoryMapper

def test_optical_memory():
    print("--- 🔬 Testing 5D Photonic Memory Mapping ---\n")
    
    # ১. চিপ মেমোরি ইনিশিয়ালাইজ করা (5x5 গ্রিড, ৩টি লেয়ার)
    mapper = GlassMemoryMapper(grid_x=5, grid_y=5, layers_z=3)
    
    # ২. একটি সাধারণ ডিজিটাল ম্যাট্রিক্স (যাতে পজিটিভ ও নেগেটিভ ডেটা আছে)
    input_matrix = [
        [ 2.5, -1.2,  3.0],
        [-4.0,  0.5, -2.2],
        [ 1.1,  5.0, -0.8]
    ]
    
    print("\n📥 Input Digital Matrix (Standard AI Weights):")
    print(np.array(input_matrix))
    
    # ৩. ডেটাকে আলোতে (Intensity এবং Phase) কনভার্ট করা
    print("\n✨ Converting to Photonic Light States...")
    intensity, phase = mapper.encode_to_light(input_matrix, z_layer=0)
    
    print("\n💡 Encoded Intensity (Brightness - Always Positive):")
    print(intensity)
    
    print("\n🌊 Encoded Phase (0 = Positive Number, 3.14 = Negative Number):")
    print(phase)
    
    # ৪. আলো থেকে আবার ডিজিটাল ভ্যালুতে ফেরত আনা
    print("\n📤 Reading from Photonic Memory (Back to Digital):")
    output_matrix = mapper.read_from_light(x_len=3, y_len=3, z_layer=0)
    print(output_matrix)
    
    # ৫. ভেরিফাই করা যে ডেটা ঠিক আছে কিনা
    if np.allclose(input_matrix, output_matrix):
        print("\n✅ SUCCESS: Data matched perfectly! Zero-loss optical conversion.")
    else:
        print("\n❌ ERROR: Data mismatch.")

if __name__ == "__main__":
    test_optical_memory()