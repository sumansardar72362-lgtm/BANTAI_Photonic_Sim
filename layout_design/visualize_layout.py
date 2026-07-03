import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_advanced_chip_layout():
    width = 5000
    height = 5000
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-500, width + 500)
    ax.set_ylim(-500, height + 500)
    
    # ১. চিপের বাইরের বডি
    chip_body = patches.Rectangle((0, 0), width, height, linewidth=3, edgecolor='black', facecolor='#e0e0e0', label='BANTAI Chip Boundary')
    ax.add_patch(chip_body)
    
    # ২. ডেটা স্টোরেজ: SRAM Memory Banks (উপরে এবং নিচে)
    # উপরের মেমরি (AI Weights স্টোর করার জন্য)
    sram_top = patches.Rectangle((500, 4000), 4000, 600, linewidth=2, edgecolor='#cc7a00', facecolor='#ffcc99', label='SRAM Memory (AI Weights)')
    ax.add_patch(sram_top)
    
    # নিচের মেমরি (Input Data স্টোর করার জন্য)
    sram_bottom = patches.Rectangle((500, 400), 4000, 600, linewidth=2, edgecolor='#cc7a00', facecolor='#ffcc99', label='SRAM Memory (Input Data)')
    ax.add_patch(sram_bottom)

    # ৩. DAC (Digital to Analog Converter) - মেমরি থেকে ডেটা লেজারে পাঠানোর জন্য
    dac_block = patches.Rectangle((600, 1200), 400, 2600, linewidth=2, edgecolor='#660066', facecolor='#e6b3ff', label='DAC (Data Converter)')
    ax.add_patch(dac_block)

    # ৪. মাঝখানের ফোটোনিক ম্যাক কোর (প্রসেসিং এরিয়া)
    mac_core = patches.Rectangle((1200, 1200), 2600, 2600, linewidth=2, edgecolor='blue', facecolor='#cce5ff', label='Photonic MAC Array (Processing)')
    ax.add_patch(mac_core)
    
    # ৫. ADC (Analog to Digital Converter) - রেজাল্ট আবার মেমরিতে সেভ করার জন্য
    adc_block = patches.Rectangle((4000, 1200), 400, 2600, linewidth=2, edgecolor='#006622', facecolor='#b3ffcc', label='ADC (Output Converter)')
    ax.add_patch(adc_block)

    # ৬. লেজার ইনপুট এবং অপটিক্যাল আউটপুট পোর্ট
    plt.scatter([0], [height/2], color='red', s=150, zorder=5, label='Laser Input')
    plt.scatter([width], [height/2], color='green', s=150, zorder=5, label='Optical Output')
    
    # আলো যাওয়ার রাস্তা (Waveguide)
    plt.plot([0, 1200], [height/2, height/2], color='red', linestyle='--', linewidth=2)
    plt.plot([3800, 5000], [height/2, height/2], color='green', linestyle='--', linewidth=2)

    # গ্রাফের ডিজাইন
    ax.set_aspect('equal')
    plt.title('BANTAI Photonic AI Chip - Full Architecture with Storage', fontsize=16, fontweight='bold')
    plt.xlabel('Width (µm)', fontsize=12)
    plt.ylabel('Height (µm)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # লিজেন্ড বা ইনডেক্স একটু বাইরে রাখছি যাতে ডিজাইন ঢেকে না যায়
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # ছবি সেভ করা
    plt.savefig('layout_design/bantai_full_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ Advanced Design with Storage saved successfully!")
    
    plt.show()

if __name__ == "__main__":
    draw_advanced_chip_layout()