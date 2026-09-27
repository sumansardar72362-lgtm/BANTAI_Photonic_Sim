import numpy as np
# C++ Tensor5D এর সাথে টাইপ চেকিংয়ের জন্য
from kecia.core.tensor5d import Tensor5D 

class Adam:
    """
    BANTAI Kecia - Adam Optimizer.
    সবচেয়ে জনপ্রিয় এবং দ্রুতগামী ডিপ লার্নিং অপটিমাইজার।
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 

        # 💥 ফিক্স: Tensor5D অবজেক্ট থেকে শেপ নিয়ে জিরো ম্যাট্রিক্স বানানো
        self.m = [np.zeros_like(p.data if hasattr(p, 'data') else p) for p in self.params]
        self.v = [np.zeros_like(p.data if hasattr(p, 'data') else p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad = None # জিরো ম্যাট্রিক্সের বদলে None করে দেওয়াটা মেমরির জন্য ভালো

    def step(self):
        self.t += 1
        
        for i, p in enumerate(self.params):
            if not hasattr(p, 'grad') or p.grad is None:
                continue

            # 💥 ফিক্স: p.grad যদি Tensor5D হয়, তবে .data বের করে নাও
            grad = p.grad.data if isinstance(p.grad, Tensor5D) else p.grad

            # 1. Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # 2. Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # 3. Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
            
            # 4. Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t))

            # 5. Update weights & biases
            # 💥 ফিক্স: update_step বের করে সেটাকে p.data থেকে বিয়োগ করা হচ্ছে
            update_step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            if isinstance(p, Tensor5D):
                p.data -= update_step
            else:
                p -= update_step