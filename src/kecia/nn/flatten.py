import numpy as np
from kecia.core.tensor5d import Tensor5D

class Flatten:
    def __init__(self):
        self.last_shape = None

    def forward(self, x):
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x)
        self.last_shape = x.shape # ⚡ ব্যাকওয়ার্ডের জন্য শেপ সেভ রাখা
        flat_data = x.data.reshape(x.shape[0], -1)
        return Tensor5D(flat_data, device=x.device)

    def backward(self, grad_output):
        """গ্রেডিয়েন্টকে আবার ইমেজের 4D শেপে ফিরিয়ে আনা"""
        if not isinstance(grad_output, Tensor5D):
            grad_output = Tensor5D(grad_output)
        reshaped_grad = grad_output.data.reshape(self.last_shape)
        return Tensor5D(reshaped_grad, device=grad_output.device)

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)