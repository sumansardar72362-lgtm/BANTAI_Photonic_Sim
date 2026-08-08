import ctypes
import os
import numpy as np

class PhotonicHardwareDriver:
    """
    BANTAI Kecia SDK - Photonic Hardware Bridge
    This class connects Python to the ultra-fast C++ native DLL.
    """
    
    def __init__(self):
        self.is_ready = False
        self.lib = None
        self._load_driver()

    def _load_driver(self):
        try:
            # src/kecia/backend/photonic_driver.py থেকে native/drivers/bantai_driver.dll এর পাথ বের করা
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            dll_path = os.path.join(base_dir, 'native', 'drivers', 'bantai_driver.dll')
            
            if not os.path.exists(dll_path):
                print("⚠️ WARNING: bantai_driver.dll not found. Falling back to CPU Simulator mode.")
                return

            # C++ DLL লোড করা
            self.lib = ctypes.CDLL(dll_path)
            
            # C++ ফাংশনের সিগনেচার (Data types) বলে দেওয়া
            self.lib.send_data_to_photonic_chip.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ]
            self.lib.send_data_to_photonic_chip.restype = None
            
            self.is_ready = True
            
        except Exception as e:
            print(f"❌ Failed to load Photonic Driver: {e}")

    def compute_optical_multiplication(self, matrix_a, matrix_b):
        """
        Takes two flat numpy arrays and processes them using the C++ Photonic DLL.
        """
        # ডেটাকে C++ এর float32 তে কনভার্ট করা হচ্ছে
        arr_a = np.array(matrix_a, dtype=np.float32).flatten()
        arr_b = np.array(matrix_b, dtype=np.float32).flatten()
        
        total_elements = len(arr_a)
        
        if not self.is_ready:
            # যদি কোনো কারণে DLL লোড না হয়, তবে সাধারণ পাইথন দিয়েই ক্যালকুলেট করবে (0.99 লস সহ)
            return arr_a * arr_b * 0.99 

        out_arr = np.zeros(total_elements, dtype=np.float32)

        # C++ মেমরি পয়েন্টার তৈরি
        ptr_a = arr_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_b = arr_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptr_out = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # ⚡ হার্ডওয়্যার ফাংশন কল (Super Fast)
        self.lib.send_data_to_photonic_chip(ptr_a, ptr_b, ptr_out, total_elements)

        return out_arr

# গ্লোবাল ড্রাইভার ইন্সট্যান্স, যাতে পুরো প্রজেক্ট থেকে একবারই কল করা যায়
hardware_bridge = PhotonicHardwareDriver()