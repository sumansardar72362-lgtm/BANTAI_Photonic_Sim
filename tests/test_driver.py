import ctypes
import os
import sys

def test_dll_connection():
    print("\n🔍 --- BANTAI NATIVE DRIVER TEST --- 🔍\n")
    
    # তোমার native/drivers ফোল্ডারের পাথ বের করা হচ্ছে
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dll_path = os.path.join(base_dir, 'native', 'drivers', 'bantai_driver.dll')
    
    print(f"📂 Looking for DLL at: {dll_path}")
    
    if not os.path.exists(dll_path):
        print("❌ Error: bantai_driver.dll file not found!")
        return

    try:
        # পাইথন ctypes ব্যবহার করে C++ DLL ফাইলটি লোড করছে
        bantai_lib = ctypes.CDLL(dll_path)
        print("✅ SUCCESS: bantai_driver.dll loaded perfectly into Python!")
        print("⚡ The Photonic Bridge is active.")
        
    except OSError as e:
        print(f"❌ OS Error loading DLL (Might be a 32-bit/64-bit mismatch or missing dependency): {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        
    print("\n---------------------------------------\n")

if __name__ == "__main__":
    test_dll_connection()