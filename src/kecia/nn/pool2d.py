import numpy as np
from kecia.core.tensor5d import Tensor5D

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.last_input = None

    def forward(self, input_tensor):
        self.last_input = input_tensor.data
        batch_size, channels, height, width = self.last_input.shape
        
        # আউটপুটের নতুন সাইজ হিসাব করা
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Max Pooling অপারেশন (ভিডিও ফ্রেমের সবচেয়ে গুরুত্বপূর্ণ পিক্সেল বের করা)
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                output[:, :, i, j] = np.max(
                    self.last_input[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)
                )
                
        return Tensor5D(output)

    def backward(self, grad_output):
        # ব্যাকপ্রোপাগেশনের সময় গ্রেডিয়েন্ট শুধু ম্যাক্সিমাম পিক্সেলগুলোতে যাবে
        batch_size, channels, height, width = self.last_input.shape
        grad_input = np.zeros_like(self.last_input)
        grad_out_data = grad_output.data
        
        _, _, out_height, out_width = grad_out_data.shape
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # ইনপুট প্যাচ বের করা
                patch = self.last_input[:, :, h_start:h_end, w_start:w_end]
                max_val = np.max(patch, axis=(2, 3), keepdims=True)
                
                # ম্যাক্সিমাম ভ্যালুর লোকেশনে মাস্ক (Mask) তৈরি করা
                mask = (patch == max_val)
                grad_input[:, :, h_start:h_end, w_start:w_end] += mask * grad_out_data[:, :, i:i+1, j:j+1]
                
        return Tensor5D(grad_input)