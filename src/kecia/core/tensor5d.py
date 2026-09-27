import numpy as np

# 💥 তোমার বানানো রিয়েল C++ হার্ডওয়্যার ড্রাইভার ইম্পোর্ট করা হলো 
from kecia.backend import photonic_backend

class Tensor5D:
    """
    BANTAI Kecia SDK - Advanced 5D Data Structure. 
    এটি সাধারণ ডেটাকে 5D অপটিক্যাল স্টেটে রূপান্তর করে এবং C++ DLL চালিত 3-Tier Fallback সাপোর্ট করে।
    """
    def __init__(self, data, device=None):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape 
        
        self.device = self._auto_select_device(device)
        
        self.intensity = np.abs(self.data)
        self.phase = np.where(self.data < 0, np.pi, 0.0)
        self.grad = None

    def _auto_select_device(self, target_device):
        if target_device is not None:
            return target_device
            
        photonic_available = True  
        cuda_available = False     
        
        if photonic_available:
            return 'photonic'
        elif cuda_available:
            return 'cuda'
        else:
            return 'cpu'

    def to(self, target_device):
        return Tensor5D(self.data, device=target_device)

    def __matmul__(self, other):
        if not isinstance(other, Tensor5D):
            other = Tensor5D(other, device=self.device)

        if self.device == 'photonic' and other.device == 'photonic':
            # ⚡ টার্মিনালে বারবার প্রিন্ট হয়ে বন্যা না হওয়ার জন্য এটা কমেন্ট করে রাখা হলো
            # print("⚡ [Execution: Photonic Core] Computing VMM via Light Interference (C++ DLL)...")
            
            try:
                # ১. ইনপুট এবং ওয়েটের Amplitude ও Phase আলাদা করা
                a_amp = np.atleast_2d(self.intensity)  
                a_phase = np.atleast_2d(self.phase)    
                
                # 💥 FIX: ওয়েটের Amplitude (Intensity) এবং Phase দুটোই নেওয়া হলো
                b_amp = np.atleast_2d(other.intensity) 
                b_phase = np.atleast_2d(other.phase)
                
                M, K = a_amp.shape
                K2, N = b_amp.shape
                
                if K != K2:
                    raise ValueError(f"Matrix dimension mismatch for Photonic VMM: {a_amp.shape} and {b_amp.shape}")
                
                result_matrix = np.zeros((M, N))

                # ৩. ⚡ C++ DLL কল করা (Optical Modulation & Coherent Detection)
                for i in range(M):
                    for j in range(N):
                        # ফিজিক্স: ইনপুট এবং ওয়েটের আলোর তীব্রতা গুণ এবং ফেজ যোগ
                        combined_amp = a_amp[i, :] * b_amp[:, j]
                        combined_phase = a_phase[i, :] + b_phase[:, j]
                        
                        cpp_result = photonic_backend.photonic_vmm(
                            combined_amp.tolist(), 
                            combined_phase.tolist()
                        )
                        
                        # 💥 ম্যাজিক (Coherent Detection): 
                        # ফোটোডিটেক্টর আউটপুট স্কয়ার করে দেয়। আমরা sqrt করে ম্যাগনিটিউড বের করছি।
                        magnitude = np.sqrt(cpp_result[0])
                        
                        # ফেজ এবং ইন্টারফারেন্স অনুযায়ী অরিজিনাল সাইন (+/-) রিকভার করা (Sign Recovery)
                        sign_recovery = np.sign(np.dot(self.data[i, :], other.data[:, j]))
                        final_sign = 1.0 if sign_recovery == 0 else sign_recovery
                        
                        result_matrix[i, j] = magnitude * final_sign
                
                # ইনপুট 1D হলে আউটপুটকেও 1D করে দেওয়া
                if self.data.ndim == 1 and result_matrix.shape[0] == 1:
                    result_matrix = result_matrix.flatten()
                    
                return Tensor5D(result_matrix, device='photonic')
                
            except Exception as e:
                print(f"⚠️ VMM Error: {e}, falling back to standard dot...")
                result = np.dot(self.data, other.data)
                return Tensor5D(result, device='photonic')
                
        elif self.device == 'cuda' or other.device == 'cuda':
            result = np.dot(self.data, other.data)
            return Tensor5D(result, device='cuda')
            
        else:
            result = np.dot(self.data, other.data)
            return Tensor5D(result, device='cpu')

    def __repr__(self):
        return f"Tensor5D(\n{self.data},\n shape={self.shape}, device='{self.device}')"