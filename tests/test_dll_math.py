import ctypes
import os
import numpy as np

def run_hardware_math():
    print("\n⚡ --- BANTAI HARDWARE MATH TEST --- ⚡\n")
    
    # 1. DLL লোড করা
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dll_path = os.path.join(base_dir, 'native', 'drivers', 'bantai_driver.dll')
    bantai_lib = ctypes.CDLL(dll_path)

    # 2. পাইথনকে C++ এর ডেটা টাইপ (Signature) বুঝিয়ে দেওয়া
    # C++ Code: void send_data_to_photonic_chip(float* a, float* b, float* out, int total)
    bantai_lib.send_data_to_photonic_chip.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # matrix_a (float pointer)
        ctypes.POINTER(ctypes.c_float),  # matrix_b (float pointer)
        ctypes.POINTER(ctypes.c_float),  # output (float pointer)
        ctypes.c_int                     # total_elements (integer)
    ]
    bantai_lib.send_data_to_photonic_chip.restype = None # void return

    # 3. টেস্ট করার জন্য কিছু ডামি ডেটা বানানো (float32 ব্যবহার করছি কারণ C++ এ float মানে 32-bit)
    total_elements = 5
    array_a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    array_b = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    output_array = np.zeros(total_elements, dtype=np.float32) # ফাঁকা আউটপুট অ্যারে

    print(f"📥 Python Input A: {array_a}")
    print(f"📥 Python Input B: {array_b}")
    print("---------------------------------------")

    # 4. পাইথনের মেমরি অ্যাড্রেস (Pointer) বের করে C++ কে দেওয়া
    ptr_a = array_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_b = array_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_out = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 5. বুম! C++ ফাংশনকে কল করা!
    bantai_lib.send_data_to_photonic_chip(ptr_a, ptr_b, ptr_out, total_elements)

    # 6. রেজাল্ট প্রিন্ট করা
    print("---------------------------------------")
    print(f"📤 Output from C++: {output_array}")
    print("\n✅ Python and C++ Photonic Bridge is fully operational!")

if __name__ == "__main__":
    run_hardware_math()