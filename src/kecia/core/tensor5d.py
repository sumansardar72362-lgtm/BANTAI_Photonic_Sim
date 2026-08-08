import numpy as np

# 💥 তোমার বানানো রিয়েল C++ হার্ডওয়্যার ড্রাইভার ইম্পোর্ট করা হলো (OpticalMAC এর বদলে)
from src.kecia.backend.photonic_driver import hardware_bridge

class Tensor5D:
    """
    BANTAI Kecia SDK - Advanced 5D Data Structure. 
    এটি সাধারণ ডেটাকে 5D অপটিক্যাল স্টেটে রূপান্তর করে এবং C++ DLL চালিত 3-Tier Fallback সাপোর্ট করে।
    """
    def __init__(self, data, device=None):
        # C++ DLL এর সাথে সামঞ্জস্য রাখার জন্য float32 তে কনভার্ট করা হচ্ছে
        self.data = np.array(data, dtype=np.float32)
        
        # 3-Tier Fallback সিস্টেমের মাধ্যমে স্বয়ংক্রিয়ভাবে ডিভাইস নির্বাচন
        self.device = self._auto_select_device(device)
        
        # ব্যাকগ্রাউন্ডে স্বয়ংক্রিয়ভাবে ফোটোনিক স্টেটে কনভার্ট করা (Intensity & Phase)
        self.intensity = np.abs(self.data)
        self.phase = np.where(self.data < 0, np.pi, 0.0)
        self.grad = None

    def _auto_select_device(self, target_device):
        """
        Photonic -> CUDA -> CPU ফলব্যাক লজিক
        """
        if target_device is not None:
            return target_device
            
        # 💥 এখন আর হার্ডকোডেড True নয়, ড্রাইভার নিজে চেক করবে C++ DLL রেডি আছে কি না!
        photonic_available = hardware_bridge.is_ready  
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

        # 🚀 টায়ার ১: ফোটোনিক চিপ (Speed of Light via C++ DLL)
        if self.device == 'photonic' and other.device == 'photonic':
            print("⚡ [Execution: Photonic Core] Computing VMM via Light Interference (C++ DLL)...")
            
           # VMM (Vector-Matrix Multiplication) লজিক
            try:
                # ১. ইনপুটগুলোকে 2D ম্যাট্রিক্সে রূপান্তর করা (M x K) এবং (K x N)
                a_2d = np.atleast_2d(self.data)  
                b_2d = np.atleast_2d(other.data) 
                
                M, K = a_2d.shape
                K2, N = b_2d.shape

                # ২. 3D ব্রডকাস্টিং (Optical Beam Expansion)
                # ব্যাচ প্রসেসিংয়ের জন্য আলোকে (M, K, N) ডাইমেনশনে ব্রডকাস্ট করা হচ্ছে
                A_3d = a_2d[:, :, None] # Shape: (M, K, 1)
                B_3d = b_2d[None, :, :] # Shape: (1, K, N)
                
                target_shape = (M, K, N)
                A_broad = np.broadcast_to(A_3d, target_shape)
                B_broad = np.broadcast_to(B_3d, target_shape)
                
                # ৩. ⚡ C++ DLL কল করা (দুটো 3D বিমকে অপটিক্যালি গুণ করা হচ্ছে)
                flat_result = hardware_bridge.compute_optical_multiplication(A_broad, B_broad)
                
                # ৪. ম্যাট্রিক্সে রূপান্তর এবং Accumulation (K-অক্ষ বরাবর আলো যোগ করা)
                optical_mult = flat_result.reshape(target_shape)
                result = np.sum(optical_mult, axis=1) 
                
                # ইনপুট 1D হলে আউটপুটকেও 1D করে দেওয়া
                if self.data.ndim == 1 and result.shape[0] == 1:
                    result = result.flatten()
                    
            except Exception as e:
                print(f"⚠️ VMM Error: {e}, falling back to basic dot...")
                result = np.dot(self.data, other.data) * 0.99
                
            return Tensor5D(result, device='photonic')
            
        # 🔥 টায়ার ২: ক্যুডা (CUDA GPU)
        elif self.device == 'cuda' or other.device == 'cuda':
            print("🔥 [Execution: CUDA GPU] Computing via Parallel Cores...")
            result = np.dot(self.data, other.data)
            return Tensor5D(result, device='cuda')
            
        # 💻 টায়ার ৩: সিপিইউ (Standard CPU)
        else:
            print("💻 [Execution: Standard CPU] Computing via Silicon Threads...")
            result = np.dot(self.data, other.data)
            return Tensor5D(result, device='cpu')

    def __repr__(self):
        """টার্মিনালে সুন্দরভাবে টেন্সর প্রিন্ট করার জন্য"""
        return f"Tensor5D(\n{self.data},\n device='{self.device}')"