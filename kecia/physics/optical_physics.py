import math

class OpticalResonatorSim:
    def __init__(self, wavelength_nm=1550, n_eff=2.0):
        # তরঙ্গদৈর্ঘ্যকে মিটারে কনভার্ট করা হচ্ছে (1550 nm = 1550 * 10^-9 m)
        self.wavelength = wavelength_nm * 1e-9 
        self.n_eff = n_eff
        print("🔬 BANTAI Optical Physics Engine Initialized...")

    def calculate_radius(self, mode_number):
        """
        রেজোন্যান্স ইকুয়েশন: 2 * pi * r * n_eff = m * lambda
        অতএব, r = (m * lambda) / (2 * pi * n_eff)
        """
        # মিটারে ব্যাসার্ধ বের করা
        radius_meters = (mode_number * self.wavelength) / (2 * math.pi * self.n_eff)
        
        # মাইক্রোমিটারে (µm) কনভার্ট করা, যাতে বুঝতে সুবিধা হয়
        radius_micrometers = radius_meters * 1e6
        return round(radius_micrometers, 4)

# টেস্টিং ব্লক
if __name__ == "__main__":
    sim = OpticalResonatorSim()
    
    # ফোটোনিক্সের ক্ষেত্রে সাধারণত 'm' (Mode number) ৪০-৫০ এর মধ্যে রাখা হয়
    m = 40 
    
    r_um = sim.calculate_radius(m)
    
    print("\n--- ⚡ Chip Physics Validation ---")
    print("Target Wavelength (Laser): 1550 nm")
    print("Effective Refractive Index (Diamond): 2.0")
    print(f"Resonance Mode (m): {m}")
    print("-----------------------------------")
    print(f"✅ Optimal Micro-ring Radius: {r_um} µm (Micro-meters)")
    print("-----------------------------------")