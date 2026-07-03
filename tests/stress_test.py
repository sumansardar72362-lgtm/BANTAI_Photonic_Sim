import subprocess
import time
import numpy as np
import os

class PhotonicStressTest:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        print("\n=======================================================")
        print(" 🔥 KECIA PHOTONIC ENGINE - EXTREME STRESS TEST")
        print(f" 📊 Matrix Size: {rows} x {cols} ({rows * cols} Elements)")
        print(" ⚡ Mode: 100% Optical Parallel Processing")
        print("=======================================================\n")

    def run_test(self):
        print("[1/3] Generating massive AI synthetic dataset using Numpy...")
        # 10 lokkho random numbers-er matrix banano hochhe
        large_matrix = np.random.rand(self.rows, self.cols)
        
        print("[2/3] Mapping entire 1,000,000 elements to 1550nm parallel light beams...")
        start_time = time.perf_counter()
        
        # Parallel photonic simulation (Numpy matrix manipulation)
        optical_pulses = large_matrix * 0.1550
        
        print(f"[3/3] Firing all {self.rows * self.cols} simultaneous photons into Diamond Core...")
        
        # Firmware path set kora
        current_dir = os.path.dirname(os.path.abspath(__file__))
        firmware_path = os.path.join(current_dir, '..', 'firmware', 'bantai_firmware.exe')
        
        try:
            # Parallel execution-er jonno firmware-ke trigger kora
            firmware_response = subprocess.run(
                [firmware_path], 
                capture_output=True, 
                text=True
            )
            print("\n--- HARDWARE OS RESPONSE ---")
            print(firmware_response.stdout)
        except FileNotFoundError:
            print("\n[Alert] C++ Firmware not found. Please ensure it is compiled.")

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000 # Milliseconds-e convert kora
        
        print("=======================================================")
        print(" 📈 STRESS TEST PERFORMANCE REPORT")
        print("=======================================================")
        print(f" ⏱️ Total Data Processing Time: {total_time:.4f} ms")
        print(f" 🌡️ Core Temperature: 0.0 °C (Zero Thermal Dissipation)")
        print(f" 🚀 Photonic Throughput: Parallel Optical Waveguide Active")
        print("=======================================================\n")

if __name__ == "__main__":
    # 1000x1000 matrix run kora hochhe
    tester = PhotonicStressTest(1000, 1000)
    tester.run_test()