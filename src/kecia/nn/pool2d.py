import numpy as np
from kecia.core.tensor5d import Tensor5D

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.last_input = None

    def forward(self, x):
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x)
        self.last_input = x
        batch_size, channels, h, w = x.shape
        out_h = h // self.kernel_size
        out_w = w // self.kernel_size
        
        reshaped = x.data.reshape(batch_size, channels, out_h, self.kernel_size, out_w, self.kernel_size)
        out_data = reshaped.max(axis=(3, 5))
        return Tensor5D(out_data, device=x.device)

    def backward(self, grad_output):
        """যে পিক্সেলগুলো সিলেক্ট হয়েছিল, গ্রেডিয়েন্ট শুধু তাদের কাছেই যাবে"""
        if not isinstance(grad_output, Tensor5D):
            grad_output = Tensor5D(grad_output)
            
        x_data = self.last_input.data
        batch_size, channels, h, w = x_data.shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        
        # খালি ফ্রেম তৈরি (যেখানে গ্রেডিয়েন্ট বসবে)
        grad_input = np.zeros_like(x_data)
        
        # সহজ Numpy ট্রিক: Kronecker product দিয়ে গ্রেডিয়েন্টকে বড় করা এবং Max Mask দিয়ে ফিল্টার করা
        grad_expanded = np.repeat(np.repeat(grad_output.data, self.kernel_size, axis=2), self.kernel_size, axis=3)
        
        # ইনপুটের ম্যাক্সিমাম ভ্যালুগুলোর পজিশন বের করে সেখানে গ্রেডিয়েন্ট বসানো
        reshaped_x = x_data.reshape(batch_size, channels, out_h, self.kernel_size, out_w, self.kernel_size)
        max_vals = reshaped_x.max(axis=(3, 5), keepdims=True)
        max_vals = np.repeat(np.repeat(max_vals, self.kernel_size, axis=3), self.kernel_size, axis=5).reshape(x_data.shape)
        
        mask = (x_data == max_vals)
        grad_input = mask * grad_expanded
        
        return Tensor5D(grad_input, device=grad_output.device)

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)