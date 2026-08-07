import numpy as np

class KeciaOptimizer:
    def __init__(self, learning_rate=0.01):
        # Learning Rate মানে হলো এআই কত দ্রুত শিখবে
        self.lr = learning_rate
        print(f"⚙️ [Kecia Optimizer] Booted up. Learning Rate: {self.lr}")

    def calculate_loss(self, true_labels, predictions):
        """Mean Squared Error (MSE) - এআই-এর ভুল কতটুকু সেটা মাপা হচ্ছে"""
        loss = np.mean(np.square(true_labels - predictions))
        print(f"📉 [Kecia Trainer] Current Loss (Error Margin): {loss:.4f}")
        return loss

    def backpropagate(self, layer, error_gradient):
        """ভুলটা শুধরানোর জন্য ফোটোনিক কোর-এ ব্যাকপ্রোপাগেশন ফায়ার করা হচ্ছে"""
        print("🔄 [Kecia Trainer] Sending Error Gradients to Photonic Core...")
        
        # এখানে C++ ফার্মওয়্যারের ডামি কানেকশন সিমুলেট করা হচ্ছে
        # ফিজিক্যাল চিপে এই গুণটা আলোর গতিতে লেজারের সাহায্যে উল্টোদিকে হবে
        weight_updates = error_gradient * self.lr
        
        # এআই-এর ব্রেইনের মেমরি (Weights) আপডেট করা হলো
        layer.weights -= weight_updates
        
        print("✅ [Kecia Trainer] Core Weights Updated at the Speed of Light!\n")