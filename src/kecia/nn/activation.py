import numpy as np
from src.kecia.core.tensor5d import Tensor5D

class ReLU5D:
    """
    Photonic Rectified Linear Unit (ReLU).
    ফোটোনিক চিপে 'থ্রেশহোল্ডিং ফিল্টার' হিসেবে কাজ করে।
    যেসব আলোর সিগন্যাল (ভ্যালু) 0 এর নিচে, তাদের ব্লক করে দেয় (Noise Reduction)।
    """
    def __init__(self):
        self.last_input = None

    def forward(self, x):
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x)
        
        # ব্যাকওয়ার্ড পাসের জন্য অরিজিনাল ইনপুট সেভ করে রাখা
        self.last_input = x.data
        
        # ম্যাথ: f(x) = max(0, x) (নেগেটিভ ভ্যালুগুলোকে 0 করে দেওয়া)
        out_data = np.maximum(0, self.last_input).astype(np.float32)
        return Tensor5D(out_data, device=x.device)

    def backward(self, grad_output):
        grad_out_data = grad_output.data if isinstance(grad_output, Tensor5D) else grad_output
        
        # Gradient ক্যালকুলেশন: ইনপুট > 0 হলে 1, নইলে 0
        relu_grad = (self.last_input > 0).astype(np.float32)
        
        # চেইন রুল (Chain Rule) অনুযায়ী পেছনের গ্রেডিয়েন্টের সাথে গুণ
        in_grad = (grad_out_data * relu_grad).astype(np.float32)
        return Tensor5D(in_grad, device=self.last_input_device(grad_output))

    def last_input_device(self, grad_output):
        return grad_output.device if isinstance(grad_output, Tensor5D) else 'photonic'

    def __call__(self, x):
        return self.forward(x)


class Sigmoid5D:
    """
    Photonic Sigmoid Activation.
    বিশাল ইনটেনসিটির আলোকে 0 থেকে 1 এর মধ্যে নরমালাইজ করে (Optical Squeezing).
    """
    def __init__(self):
        self.last_output = None

    def forward(self, x):
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x)
            
        # ওভারফ্লো (Overflow) ঠেকানোর জন্য ভ্যালুগুলোকে -50 থেকে 50 এর মধ্যে ক্লিপ করা
        x_clipped = np.clip(x.data, -50, 50) 
        
        # ম্যাথ: f(x) = 1 / (1 + e^-x)
        out_data = (1.0 / (1.0 + np.exp(-x_clipped))).astype(np.float32)
        
        # ব্যাকওয়ার্ড পাসের জন্য আউটপুট সেভ করে রাখা
        self.last_output = out_data
        return Tensor5D(out_data, device=x.device)

    def backward(self, grad_output):
        grad_out_data = grad_output.data if isinstance(grad_output, Tensor5D) else grad_output
        
        # Sigmoid এর ডেরিভেটিভ ফর্মুলা: sigmoid(x) * (1 - sigmoid(x))
        sig_grad = self.last_output * (1.0 - self.last_output)
        
        # চেইন রুল
        in_grad = (grad_out_data * sig_grad).astype(np.float32)
        return Tensor5D(in_grad, device=self.last_input_device(grad_output))
        
    def last_input_device(self, grad_output):
        return grad_output.device if isinstance(grad_output, Tensor5D) else 'photonic'

    def __call__(self, x):
        return self.forward(x)