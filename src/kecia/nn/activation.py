import numpy as np
from src.kecia.core.tensor5d import Tensor5D

class PhotonicReLU:
    """Optical Thresholding Filter for Photonic Chip (Simulates ReLU)"""
    
    def __init__(self):
        # ব্যাকপ্রপাগেশনের ইনপুট ধরে রাখার জন্য ভেরিয়েবল
        self.last_input = None 
        
    def __call__(self, tensor: Tensor5D) -> Tensor5D:
        """লেয়ারকে ফাংশনের মতো কল করার জন্য (যেমন: relu(x))"""
        # ব্যাকপ্রপাগেশনের জন্য ইনপুট সেভ করা হচ্ছে
        self.last_input = tensor 
        
        if tensor.device == 'cpu':
            out_data = np.maximum(0, tensor.data)
            return Tensor5D(out_data, device='cpu')
            
        elif tensor.device == 'photonic':
            out_data = np.maximum(0, tensor.data)
            out_data = np.round(out_data, decimals=4) 
            return Tensor5D(out_data, device='photonic')

    def backward(self, grad_output):
        """
        ReLU এর ডেরিভেটিভ: ইনপুট > ০ হলে গ্রেডিয়েন্ট ১, নাহলে ০।
        """
        grad_out_data = grad_output.data if isinstance(grad_output, Tensor5D) else grad_output
        
        # যেখানে ইনপুট পজিটিভ ছিল, শুধু সেখানেই ভুলের সিগন্যাল পাস হবে
        relu_grad = np.where(self.last_input.data > 0, 1.0, 0.0)
        final_grad = grad_out_data * relu_grad
        
        return Tensor5D(final_grad, device=grad_output.device)