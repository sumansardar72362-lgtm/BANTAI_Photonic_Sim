import sys
import os
import numpy as np

# মেইন ফোল্ডার থেকে মডিউল ইম্পোর্ট করার পাথ সেটআপ
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kecia.core.glass_memory import GlassMemoryMapper

def test_save_load():
    print("--- 💾 Testing 5D Photonic Memory Save/Load ---\n")
    
    # ১. প্রথম চিপ ইনিশিয়ালাইজ করা
    mapper_original = GlassMemoryMapper(grid_x=5, grid_y=5, layers_z=3)
    
    # ২. কিছু টেস্ট ডেটা আলোতে কনভার্ট করে মেমোরিতে রাখা
    input_matrix = [
        [ 1.5, -2.2,  3.1],
        [-4.5,  0.8, -1.9],
        [ 2.1,  4.0, -0.3]
    ]
    print("\n✨ Encoding data to Photonic Light States...")
    mapper_original.encode_to_light(input_matrix, z_layer=0)
    
    # ৩. মেমোরি সেভ করা (models ফোল্ডারের ভেতর)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model_path = os.path.join(model_dir, 'photonic_core.kecia')
    
    print("\n--- Saving Current Optical State ---")
    mapper_original.save_state(model_path)
    
    # ৪. পাওয়ার অফ সিমুলেশন (পুরোনো চিপ বাদ দিয়ে একদম নতুন ফাঁকা চিপ নেওয়া হলো)
    print("\n--- Simulating Power Off (Creating a new empty chip) ---")
    mapper_new = GlassMemoryMapper(grid_x=5, grid_y=5, layers_z=3)
    
    # ৫. নতুন চিপে পুরোনো সেভ করা ফাইল থেকে মেমোরি লোড করা
    print("\n--- Reloading AI State from .kecia file ---")
    mapper_new.load_state(model_path)
    
    # ৬. নতুন চিপ থেকে ডেটা পড়ে চেক করা
    print("\n📤 Reading from Restored Photonic Memory:")
    restored_matrix = mapper_new.read_from_light(x_len=3, y_len=3, z_layer=0)
    print(restored_matrix)
    
    # ৭. ভেরিফিকেশন
    if np.allclose(input_matrix, restored_matrix):
        print("\n✅ SUCCESS: Memory Save/Load works perfectly! Data restored with zero loss.")
    else:
        print("\n❌ ERROR: Data mismatch after restore.")

if __name__ == "__main__":
    test_save_load()