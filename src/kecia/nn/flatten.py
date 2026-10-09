import numpy as np
from kecia.core.tensor5d import Tensor5D

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.data.shape
        batch_size = self.input_shape[0]
        
        # পিক্সেলগুলোকে চ্যাপ্টা (Flatten) করে এক লাইনে সাজানো হচ্ছে
        flattened = input_tensor.data.reshape(batch_size, -1)
        return Tensor5D(flattened)

    def backward(self, grad_output):
        # ব্যাকপ্রোপাগেশনের সময় আবার আগের ভিডিও ফ্রেমে (4D) ফেরত পাঠানো হচ্ছে
        grad_input = grad_output.data.reshape(self.input_shape)
        return Tensor5D(grad_input)