class SGD:
    """
    Stochastic Gradient Descent (SGD) Optimizer.
    Loss Function থেকে পাওয়া ভুলের (Gradient) ওপর ভিত্তি করে মডেলের ওয়েইট আপডেট করে।
    """
    def __init__(self, parameters, lr=0.01):
        # parameters হলো মডেলের ওয়েইট এবং বায়াস এর লিস্ট
        self.parameters = parameters
        self.lr = lr  # Learning Rate: মডেল কত দ্রুত বা ধীরে শিখবে

    def step(self):
        """
        ওয়েইট আপডেট করার মূল ফাংশন। 
        Formula: Weight = Weight - (Learning_Rate * Gradient)
        """
        for param in self.parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                # ⚡ ফোটোনিক চিপের ওয়েইট (Intensity/Phase) আপডেট হচ্ছে
                param.data -= self.lr * param.grad.data
                
                # ফোটোনিক স্টেটে নতুন ভ্যালুর জন্য Intensity ও Phase রি-ক্যালকুলেট করা
                import numpy as np
                param.intensity = np.abs(param.data)
                param.phase = np.where(param.data < 0, np.pi, 0.0)

    def zero_grad(self):
        """
        প্রতিবার নতুন করে শেখার আগে আগের ভুলগুলোর (gradients) মেমোরি ক্লিয়ার করে দেওয়া।
        """
        for param in self.parameters:
            if hasattr(param, 'grad'):
                param.grad = None