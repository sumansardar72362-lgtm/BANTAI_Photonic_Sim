import numpy as np
from kecia.core.tensor5d import Tensor5D

def im2col(image, kernel_size, stride=1):
    """ইমেজকে (Image) কলামে (Column) কনভার্ট করার জাদুকরী অ্যালগরিদম"""
    batch_size, channels, height, width = image.shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1

    col = np.zeros((batch_size, channels, kernel_size, kernel_size, out_h, out_w))

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = image[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
    return col, out_h, out_w

def col2im(col, input_shape, kernel_size, stride=1):
    """কলাম ম্যাট্রিক্সকে আবার অরিজিনাল ইমেজে ফেরত নেওয়ার ম্যাথ (Backward এর জন্য)"""
    batch_size, channels, height, width = input_shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    
    img = np.zeros(input_shape)
    col = col.reshape(batch_size, out_h, out_w, channels, kernel_size, kernel_size).transpose(0, 3, 4, 5, 1, 2)
    
    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        print(f"👁️ Photonic Conv2D Ready: {in_channels} In -> {out_channels} Out")

        fan_in = in_channels * kernel_size * kernel_size
        self.filters = Tensor5D(np.random.randn(out_channels, fan_in) * (np.sqrt(2.0 / fan_in)))
        self.bias = Tensor5D(np.zeros((out_channels, 1)))

        # ব্যাকওয়ার্ডের জন্য ভেরিয়েবল
        self.last_input = None
        self.col = None

    def forward(self, x):
        if not isinstance(x, Tensor5D):
            x = Tensor5D(x)
            
        self.last_input = x  # ⚡ ফিক্স: ব্যাকওয়ার্ডের জন্য সেভ রাখা হলো
        batch_size = x.shape[0]

        # ১. Im2Col (ইমেজকে ম্যাট্রিক্সে রূপান্তর)
        col, out_h, out_w = im2col(x.data, self.kernel_size, self.stride)
        self.col = col       # ⚡ ফিক্স: ব্যাকওয়ার্ডের জন্য সেভ রাখা হলো

        # ২. ⚡ Photonic VMM Magic
        col_tensor = Tensor5D(col, device=x.device)
        weight_tensor = Tensor5D(self.filters.data.T, device=x.device) 
        
        out_tensor = col_tensor @ weight_tensor 

        # ৩. বায়াস যোগ এবং রিশেপ করে আবার 4D ইমেজে রূপান্তর
        out_data = out_tensor.data.T + self.bias.data
        out_data = out_data.reshape(self.out_channels, batch_size, out_h, out_w).transpose(1, 0, 2, 3)

        return Tensor5D(out_data, device=x.device)

    def backward(self, grad_output):
        if not isinstance(grad_output, Tensor5D):
            grad_output = Tensor5D(grad_output)
            
        batch_size = self.last_input.shape[0]
        
        # ১. গ্রেডিয়েন্টকে ম্যাট্রিক্সে রিশেপ করা
        grad_reshaped = grad_output.data.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        
        # ২. ⚡ ফিল্টারের (Weights) গ্রেডিয়েন্ট: grad_reshaped @ self.col
        # grad_reshaped হলো (16, 1800) আর self.col হলো (1800, 27)
        col_tensor = Tensor5D(self.col, device=grad_output.device) # 👈 .T রিমুভ করা হলো
        grad_reshaped_tensor = Tensor5D(grad_reshaped, device=grad_output.device)
        
        grad_weights = grad_reshaped_tensor @ col_tensor # আউটপুট: (16, 27)
        self.filters.grad = grad_weights 
        
        # বায়াসের গ্রেডিয়েন্ট
        self.bias.grad = Tensor5D(np.sum(grad_reshaped, axis=1, keepdims=True))
        
        # ৩. ইনপুটের গ্রেডিয়েন্ট: grad_reshaped.T @ weights
        # grad_reshaped.T হলো (1800, 16) আর weights হলো (16, 27)
        grad_reshaped_t_tensor = Tensor5D(grad_reshaped.T, device=grad_output.device)
        weights_tensor = Tensor5D(self.filters.data, device=grad_output.device)
        
        grad_col = grad_reshaped_t_tensor @ weights_tensor # আউটপুট: (1800, 27)
        
        # ৪. col2im দিয়ে ম্যাট্রিক্সকে আবার ইমেজে রূপান্তর
        grad_input = col2im(grad_col.data, self.last_input.shape, self.kernel_size, self.stride)
        
        return Tensor5D(grad_input, device=grad_output.device)

    def parameters(self):
        return [self.filters, self.bias]

    def __call__(self, x):
        return self.forward(x)