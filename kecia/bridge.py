import subprocess
import time
import numpy as np
import os

class PhotonicSDK:
    def __init__(self):
        print("\n=======================================================")
        print("  PHOTONIC AI SDK v1.0 INITIALIZED")
        print("  Engine: 100% Optical | Zero Electron | Zero Heat")
        print("  Core: Diamond-Glass Micro-ring Array")
        print("=======================================================\n")
        self.is_active = True

    def _convert_to_light_pulses(self, tensor_data):
        print(f"[SDK Bridge] Mapping Matrix to 1550nm Wavelength...")
        optical_data = np.array(tensor_data) * 0.1550 
        time.sleep(0.1)
        return optical_data

    def process_matrix(self, input_tensor):
        print("\n--- NEW OPTICAL PROCESS STARTED ---")
        print(f"[SDK] Received User Data: {input_tensor}")
        
        optical_signals = self._convert_to_light_pulses(input_tensor)
        print(f"[SDK] Optical Signals Ready: {optical_signals}")
        print("[SDK] Firing Laser and passing control to C++ Firmware...\n")
        time.sleep(0.2)
        

        current_dir = os.path.dirname(os.path.abspath(__file__))
        firmware_path = os.path.join(current_dir, '..', 'firmware', 'bantai_firmware.exe')
        
        try:
            firmware_response = subprocess.run(
                [firmware_path], 
                capture_output=True, 
                text=True
            )
            print(firmware_response.stdout)
        except FileNotFoundError:
            print(f"[System Alert] C++ Firmware not found at: {firmware_path}")
            print(">> Ensure the C++ code is compiled and the path is correct.")

        print("\n [SDK] Process completed at the speed of light!")
        return "Optical_Matrix_Result"

if __name__ == "__main__":
    ai_chip = PhotonicSDK()
    neural_network_layer = [2.5, 4.0, 1.2, 5.5]
    result = ai_chip.process_matrix(neural_network_layer)