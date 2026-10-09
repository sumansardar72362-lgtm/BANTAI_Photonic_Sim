import numpy as np
from kecia.core.tensor5d import Tensor5D

class Dropout:
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
        self.is_training = True  # ট্রেনিংয়ের সময় চালু থাকবে, প্রেডিকশনের সময় বন্ধ

    def forward(self, input_tensor):
        if not self.is_training or self.drop_rate == 0:
            return input_tensor
            
        # র‍্যান্ডমলি কিছু নিউরন বন্ধ (Drop) করে দেওয়া হচ্ছে
        keep_prob = 1.0 - self.drop_rate
        self.mask = np.random.binomial(1, keep_prob, size=input_tensor.data.shape) / keep_prob
        
        output = input_tensor.data * self.mask
        return Tensor5D(output)

    def backward(self, grad_output):
        if not self.is_training or self.drop_rate == 0:
            return grad_output
            
        grad_input = grad_output.data * self.mask
        return Tensor5D(grad_input)