import numpy as np
from kecia.core.tensor5d import Tensor5D

class MSELoss:
    """
    Mean Squared Error (MSE) Loss.
    মডেলের প্রেডিকশন এবং আসল টার্গেটের মধ্যে ভুলের পরিমাণ হিসাব করে।
    """
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions: Tensor5D, targets: Tensor5D):
        # ব্যাকপ্রপাগেশনের জন্য ডেটা সেভ করে রাখা হচ্ছে
        self.predictions = predictions
        self.targets = targets
        
        # (Prediction - Target)^2 / N
        error = predictions.data - targets.data
        loss_data = np.mean(np.square(error))
        
        return Tensor5D(loss_data, device=predictions.device)

    def backward(self):
        """
        Gradient বা ভুলের ডিরেকশন বের করে, যাতে মডেল বুঝতে পারে তাকে কোন দিকে ওয়েইট বদলাতে হবে।
        Formula: 2 * (Prediction - Target) / N
        """
        N = self.predictions.data.shape[0] if len(self.predictions.data.shape) > 0 else 1
        grad_data = 2 * (self.predictions.data - self.targets.data) / N
        
        # গ্রেডিয়েন্টও একটি Tensor5D হিসেবে রিটার্ন হবে
        return Tensor5D(grad_data, device=self.predictions.device)
        
    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)