import numpy as np
from src.kecia.physics.optical_mac import OpticalMAC

class Tensor5D:
    """
    Kecia SDK-এর মূল ডেটা স্ট্রাকচার। 
    এটি সাধারণ ডেটাকে 5D অপটিক্যাল স্টেটে রূপান্তর করে এবং 3-Tier Fallback সাপোর্ট করে।
    """
    def __init__(self, data, device=None):
        self.data = np.array(data)
        
        # 3-Tier Fallback সিস্টেমের মাধ্যমে স্বয়ংক্রিয়ভাবে ডিভাইস নির্বাচন
        self.device = self._auto_select_device(device)
        
        # ব্যাকগ্রাউন্ডে স্বয়ংক্রিয়ভাবে ফোটোনিক স্টেটে কনভার্ট করা (Intensity & Phase)
        self.intensity = np.abs(self.data)
        self.phase = np.where(self.data < 0, np.pi, 0.0)
        self.grad = None

    def _auto_select_device(self, target_device):
        """
        Photonic -> CUDA -> CPU ফলব্যাক লজিক
        """
        if target_device is not None:
            return target_device
            
        # সিমুলেশনের জন্য আমরা ধরে নিচ্ছি আমাদের চিপ কানেক্টেড আছে
        photonic_available = True  
        cuda_available = False     
        
        if photonic_available:
            return 'photonic'
        elif cuda_available:
            return 'cuda'
        else:
            return 'cpu'

    def to(self, target_device):
        """PyTorch স্টাইলে ডিভাইস সুইচ করার মেথড (.to('cuda'), .to('cpu'))"""
        return Tensor5D(self.data, device=target_device)

    def __matmul__(self, other):
        """
        পাইথনের '@' (ম্যাট্রিক্স মাল্টিপ্লিকেশন) অপারেটর ওভারলোড করা।
        এটি ডিভাইসের ওপর ভিত্তি করে ক্যালকুলেশন ইঞ্জিন সিলেক্ট করে।
        """
        if not isinstance(other, Tensor5D):
            other = Tensor5D(other, device=self.device)

        # 🚀 টায়ার ১: ফোটোনিক চিপ (Speed of Light)
        if self.device == 'photonic' and other.device == 'photonic':
            print("⚡ [Execution: Photonic Core] Computing via Light Interference...")
            mac = OpticalMAC()
            # self (1D Input) @ other (2D Weights)
            result = mac.compute_vmm(self.intensity, self.phase, other.intensity, other.phase)
            return Tensor5D(result, device='photonic')
            
        # 🔥 টায়ার ২: ক্যুডা (CUDA GPU)
        elif self.device == 'cuda' or other.device == 'cuda':
            print("🔥 [Execution: CUDA GPU] Computing via Parallel Cores...")
            result = np.dot(self.data, other.data)
            return Tensor5D(result, device='cuda')
            
        # 💻 টায়ার ৩: সিপিইউ (Standard CPU)
        else:
            print("💻 [Execution: Standard CPU] Computing via Silicon Threads...")
            result = np.dot(self.data, other.data)
            return Tensor5D(result, device='cpu')

    def __repr__(self):
        """টার্মিনালে সুন্দরভাবে টেন্সর প্রিন্ট করার জন্য"""
        return f"Tensor5D(\n{self.data},\n device='{self.device}')"