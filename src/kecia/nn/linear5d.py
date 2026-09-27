import numpy as np
from kecia.core.tensor5d import Tensor5D

class Linear5D:
    """
    BANTAI Kecia - Photonic Fully Connected Layer.
    এটি C++ ফোটোনিক কোরের মাধ্যমে ফরোয়ার্ড পাস এবং ব্যাকপ্রোপাগেশন সম্পন্ন করে।
    """
    def __init__(self, in_features, out_features, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        print(f"⚙️ Linear5D Layer Initialized: {in_features} -> {out_features}")
        
        # ওয়েট (Weights) ইনিশিয়ালাইজ করা হচ্ছে (Xavier/He স্টাইলে ছোট ভ্যালু)
        w_data = np.random.randn(in_features, out_features) * 0.1
        self.weights = Tensor5D(w_data)
        
        # বায়াস (Bias) ইনিশিয়ালাইজ
        if self.use_bias:
            b_data = np.zeros((1, out_features))
            self.bias = Tensor5D(b_data)
        else:
            self.bias = None
            
        # ব্যাকওয়ার্ড পাসের জন্য ডেটা সেভ রাখার ভেরিয়েবল
        self.last_input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x)
            
        self.last_input = x
        
        # ⚡ ফরোয়ার্ড পাস: C++ ফোটোনিক ইঞ্জিন দিয়ে VMM (Vector-Matrix Multiplication)
        out = x @ self.weights
        
        # বায়াস যোগ করা
        if self.use_bias:
            out_data = out.data + self.bias.data
            out = Tensor5D(out_data, device=out.device)
            
        return out

    def backward(self, grad_output):
        """ব্যাকপ্রোপাগেশন: ভুল শুধরানোর জন্য গ্রেডিয়েন্ট ক্যালকুলেট করা"""
        if not isinstance(grad_output, Tensor5D):
            grad_output = Tensor5D(grad_output)
            
        # ১. ওয়েটের গ্রেডিয়েন্ট: x_t @ grad_output
        x_t = Tensor5D(self.last_input.data.T)
        # ⚡ UPDATE: তোমার SGD এর সাথে মিলিয়ে .grad এ সেভ করা হচ্ছে
        self.weights.grad = x_t @ grad_output 
        
        # ২. বায়াসের গ্রেডিয়েন্ট
        if self.use_bias:
            sum_grad = np.sum(grad_output.data, axis=0, keepdims=True)
            # ⚡ UPDATE: তোমার SGD এর সাথে মিলিয়ে .grad এ সেভ করা হচ্ছে
            self.bias.grad = Tensor5D(sum_grad) 
            
        # ৩. ইনপুটের গ্রেডিয়েন্ট
        w_t = Tensor5D(self.weights.data.T)
        grad_input = grad_output @ w_t
        
        return grad_input
        
    def parameters(self):
        """SGD অপটিমাইজারকে ওয়েট এবং বায়াস দেওয়ার জন্য এই ফাংশনটা যোগ করা হলো"""
        if self.use_bias:
            return [self.weights, self.bias]
        return [self.weights]

    def __call__(self, x):
        return self.forward(x)