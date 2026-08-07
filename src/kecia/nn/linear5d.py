import numpy as np
from src.kecia.core.tensor5d import Tensor5D

class Linear5D:
    """
    Photonic Fully Connected Layer (Dense Layer).
    PyTorch-এর nn.Linear এর মতো কাজ করে, কিন্তু ফোটোনিক হার্ডওয়্যার ব্যাকএন্ড ব্যবহার করে।
    Formula: y = xA + b
    """
    def __init__(self, in_features, out_features, bias=True, device='photonic'):
        self.in_features = in_features
        self.out_features = out_features
        
        # PyTorch স্টাইলে ওয়েইট ইনিশিয়ালাইজেশন (Xavier/Kaiming এর সিম্পল ভার্সন)
        # Random weights generated and immediately stored as a Photonic Tensor5D
        weight_data = np.random.randn(in_features, out_features) * 0.1
        self.weights = Tensor5D(weight_data, device=device)
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor5D(bias_data, device=device)
        else:
            self.bias = None

    def forward(self, x):
        """ফরোয়ার্ড পাস। ইনপুট টেন্সরকে ওয়েইটের সাথে গুণ করে বায়াস যোগ করে।"""
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x, device=self.weights.device)
            
        # 🛠️ FIX: ব্যাকপ্রপাগেশনের জন্য ইনপুট সেভ করে রাখা হচ্ছে
        self.last_input = x 
            
        out = x @ self.weights
        if self.bias is not None:
            out_data = out.data + self.bias.data
            out = Tensor5D(out_data, device=out.device)
            
        return out

    def backward(self, grad_output):
        """
        ব্যাকপ্রপাগেশন। সামনের লেয়ার থেকে পাওয়া ভুলের (gradient) ওপর ভিত্তি করে 
        নিজের ওয়েইট ও বায়াসের ভুল বের করে এবং পেছনের লেয়ারের জন্য gradient পাঠায়।
        """
        input_data = self.last_input.data
        grad_out_data = grad_output.data if isinstance(grad_output, Tensor5D) else grad_output

        # ম্যাট্রিক্সের ডাইমেনশন ঠিক রাখার জন্য (1D কে 2D তে কনভার্ট)
        in_reshaped = np.atleast_2d(input_data)
        grad_reshaped = np.atleast_2d(grad_out_data)

        # 1. Weights Gradient = Input^T * Grad_Output
        w_grad = in_reshaped.T @ grad_reshaped
        self.weights.grad = Tensor5D(w_grad, device=self.weights.device)

        # 2. Bias Gradient = Sum of Grad_Output
        if self.bias is not None:
            b_grad = np.sum(grad_reshaped, axis=0)
            self.bias.grad = Tensor5D(b_grad, device=self.bias.device)

        # 3. Input Gradient (পেছনের লেয়ারের জন্য) = Grad_Output * Weights^T
        in_grad = grad_reshaped @ self.weights.data.T
        
        # ইনপুট যদি 1D ছিল, তবে রিটার্নও 1D করতে হবে
        if len(input_data.shape) == 1:
            in_grad = in_grad.flatten()
            
        return Tensor5D(in_grad, device=self.weights.device)

    def __call__(self, x):
        """Python-এর callable অবজেক্ট তৈরি করা (layer(x) কল করার জন্য)"""
        return self.forward(x)
        
    def to(self, device):
        """লেয়ারের সমস্ত প্যারামিটার অন্য ডিভাইসে (যেমন cpu) ট্রান্সফার করা"""
        self.weights = self.weights.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self

    def __repr__(self):
        return f"Linear5D(in_features={self.in_features}, out_features={self.out_features}, device='{self.weights.device}')"