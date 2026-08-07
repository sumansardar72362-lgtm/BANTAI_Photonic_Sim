import sys
import os
import numpy as np

# মেইন ফোল্ডার থেকে মডিউল ইম্পোর্ট করার পাথ সেটআপ
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kecia.core.glass_memory import GlassMemoryMapper
from src.kecia.physics.optical_mac import OpticalMAC

def test_optical_mac():
    print("--- ⚡ Testing Speed-of-Light Optical MAC Unit ---\n")
    
    # ১. একটি ডামি ওয়েইট ম্যাট্রিক্স (৩x৩) মেমোরিতে সেট করা
    weights_digital = np.array([
        [ 2.0, -1.0,  0.5],
        [-3.0,  4.0, -2.0],
        [ 1.5,  0.0,  3.0]
    ])
    
    mapper = GlassMemoryMapper(grid_x=5, grid_y=5, layers_z=3)
    mapper.encode_to_light(weights_digital, z_layer=0)
    
    # মেমোরি থেকে আলোর ওয়েইটগুলো (Intensity & Phase) বের করে আনা
    weight_intensity = mapper.optical_grid[:3, :3, 0, 0]
    weight_phase = mapper.optical_grid[:3, :3, 0, 1]

    # ২. একটি ডামি ইনপুট ভেক্টর তৈরি করা (ধরি, নিউরাল নেটওয়ার্ক থেকে আসছে)
    input_digital = np.array([1.5, -2.0, 3.0])
    
    # ইনপুটকেও আলোতে কনভার্ট করতে হবে (শুধু ম্যাথ দিয়ে সিমুলেট করছি)
    input_intensity = np.abs(input_digital)
    input_phase = np.where(input_digital < 0, np.pi, 0.0)

    print("📥 Input Vector (Digital):", input_digital)
    print("💾 Weight Matrix (Digital):\n", weights_digital)
    
    # ৩. Optical MAC ইউনিটে পাঠানো
    mac_unit = OpticalMAC()
    print("\n✨ Computing MAC via Light Interference...")
    
    optical_output = mac_unit.compute_vmm(
        input_intensity, input_phase, 
        weight_intensity, weight_phase
    )
    
    # ৪. সাধারণ CPU দিয়ে গুণ করে চেক করা (ভেরিফিকেশন)
    cpu_output = np.dot(input_digital, weights_digital)
    
    print("\n🚀 Output from Photonic Chip:", optical_output)
    print("💻 Output from Standard CPU: ", cpu_output)
    
    # ৫. রেজাল্ট ম্যাচ করছে কিনা যাচাই
    if np.allclose(optical_output, cpu_output):
        print("\n✅ SUCCESS: Optical MAC matches CPU perfectly! Interference math is flawless.")
    else:
        print("\n❌ ERROR: Outputs do not match.")

if __name__ == "__main__":
    test_optical_mac()