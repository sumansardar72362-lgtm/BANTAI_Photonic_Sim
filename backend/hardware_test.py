import ctypes
import os

class BantaiHardwareAPI:
    def __init__(self):
        print("🚀 Loading BANTAI C++ Hardware Driver...")
        
        # উইন্ডোজের DLL ফাইলটির লোকেশন বের করা
        lib_path = os.path.abspath("hardware_drivers/bantai_driver.dll")
        
        try:
            self.driver = ctypes.CDLL(lib_path)
            print("✅ C++ Driver Connected Successfully via MMIO!\n")
        except OSError:
            print("❌ Error: Driver DLL not found. Compilation failed?")
            return

        # C++ ফাংশনের ইনপুট-আউটপুট টাইপ (Memory Pointers) সেট করা
        self.driver.send_data_to_photonic_chip.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

    def run_fast_matmul(self, array_a, array_b):
        size = len(array_a)
        
        # Python ডেটাকে C++ Memory Pointer-এ কনভার্ট করা (Zero-copy)
        FloatArray = ctypes.c_float * size
        c_array_a = FloatArray(*array_a)
        c_array_b = FloatArray(*array_b)
        c_output = FloatArray(*[0.0]*size)

        # C++ ড্রাইভার কল করা (ডেটা চিপে পাঠানো হচ্ছে)
        self.driver.send_data_to_photonic_chip(c_array_a, c_array_b, c_output, size)

        # C++ থেকে রেজাল্ট পাইথনে ফেরত আনা
        return list(c_output)

# টেস্টিং ব্লক
if __name__ == "__main__":
    hw = BantaiHardwareAPI()
    
    # এআই মডেলের ডেটা (ফ্ল্যাট ম্যাট্রিক্স হিসেবে)
    data_A = [1.0, 2.0, 3.0, 4.0]
    data_B = [5.0, 6.0, 7.0, 8.0]
    
    # C++ ড্রাইভারের মাধ্যমে প্রসেস করা
    result = hw.run_fast_matmul(data_A, data_B)
    
    print("\n🎯 Final Hardware Output in Python:")
    for val in result:
        print(round(val, 4))