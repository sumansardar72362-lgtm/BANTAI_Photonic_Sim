class BantaiChipLayout:
    def __init__(self, width_mm=5, height_mm=5):
        # মিলিমিটারকে মাইক্রোমিটারে (µm) কনভার্ট করা হচ্ছে (ফ্যাক্টরিতে µm ব্যবহার হয়)
        self.width_um = width_mm * 1000
        self.height_um = height_mm * 1000
        self.components = {}
        print(f"📐 BANTAI Chip Blueprint Initialized ({width_mm}mm x {height_mm}mm)")

    def map_laser_io(self):
        """লেজার সোর্স এবং অপটিক্যাল আউটপুট পোর্টের জায়গা নির্ধারণ"""
        # চিপের একদম বাঁ-দিকের মাঝখানে লেজার ঢুকবে
        self.components['Laser_Input_Port'] = (0, self.height_um // 2)
        # চিপের ডানদিকের মাঝখান দিয়ে প্রসেস হওয়া আলো বেরোবে
        self.components['Optical_Output_Port'] = (self.width_um, self.height_um // 2)

    def map_systolic_array(self, rows, cols):
        """চিপের মাঝখানে MAC ইউনিটের বিশাল গ্রিড বসানো"""
        # চিপের সীমানা থেকে ১০% জায়গা ফাঁকা রেখে কোর বসানো শুরু হবে
        start_x = self.width_um * 0.1
        start_y = self.height_um * 0.1
        end_x = self.width_um * 0.9
        end_y = self.height_um * 0.9
        
        self.components['Systolic_Array_Start'] = (start_x, start_y)
        self.components['Systolic_Array_End'] = (end_x, end_y)
        self.components['Total_Optical_Cores'] = rows * cols

    def generate_gds_summary(self):
        """প্যাটেন্ট পেপারের জন্য লেআউট সামারি প্রিন্ট করা"""
        print("\n" + "="*50)
        print(" 📄 BANTAI CHIP GDSII FLOORPLAN SUMMARY (PATENT DRAFT)")
        print("="*50)
        print(f"Total Chip Area: {self.width_um} µm x {self.height_um} µm")
        print("-" * 50)
        
        for name, coords in self.components.items():
            if type(coords) == tuple:
                print(f"📍 {name}: X={coords[0]} µm, Y={coords[1]} µm")
            else:
                print(f"⚙️ {name}: {coords} Units")
                
        print("="*50 + "\n")


# এক্সিকিউশন ব্লক
if __name__ == "__main__":
    # ৫x৫ মিলিমিটার চিপ তৈরি হচ্ছে
    chip = BantaiChipLayout(width_mm=5, height_mm=5)
    
    # ইনপুট/আউটপুট পোর্ট ম্যাপ করা
    chip.map_laser_io()
    
    # সিস্টোলিক অ্যারে ম্যাপ করা (ধরে নিচ্ছি ২ কোটি কোরের গ্রিড)
    chip.map_systolic_array(rows=4000, cols=5000)
    
    # প্যাটেন্ট রিপোর্ট প্রিন্ট করা
    chip.generate_gds_summary()