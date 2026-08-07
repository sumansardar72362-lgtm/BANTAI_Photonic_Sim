import numpy as np

class OpticalMAC:
    """
    Optical Multiply-Accumulate (MAC) Unit.
    Simulates mathematical interference of light waves.
    Now supports Batched Matrix Multiplication (GEMM).
    """
    def __init__(self):
        pass
        
    def compute_vmm(self, in_int, in_phase, w_int, w_phase):
        """
        Batched Optical Interference Simulation.
        """
        # ১. আলো থেকে আসল এনালগ সিগন্যাল রিকভার করা (Intensity & Phase)
        in_signal = in_int * np.cos(in_phase)
        w_signal = w_int * np.cos(w_phase)
        
        # ২. ⚡ অপটিক্যাল ইন্টারফারেন্স (Batched Dot Product)
        out_signal = np.dot(in_signal, w_signal)
        
        # ৩. এনালগ হার্ডওয়্যার সিমুলেশনের জন্য সামান্য প্রিসিশন লস (Rounding)
        out_signal = np.round(out_signal, decimals=4)
        
        return out_signal