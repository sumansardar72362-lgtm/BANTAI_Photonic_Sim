import sys
import os
import numpy as np

# মেইন প্রজেক্টের পাথ সেটআপ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kecia.core.tensor5d import Tensor5D
from src.kecia.nn.linear5d import Linear5D
from src.kecia.nn.loss import MSELoss
from src.kecia.optim.sgd import SGD

def train_photonic_network():
    print("\n🧠 --- BANTAI AI TRAINING TEST (C++ PHOTONIC DRIVER) --- 🧠\n")

    # ১. সিম্পল ডেটাসেট (Task: ইনপুটকে দ্বিগুণ করা)
    # Input: 1, 2, 3 -> Target: 2, 4, 6
    X_data = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    Y_data = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)
    
    X = Tensor5D(X_data)
    Y = Tensor5D(Y_data)

    # ২. মডেল ডিজাইন (১টি ইনপুট, ১টি আউটপুট)
    layer = Linear5D(in_features=1, out_features=1)

    # ৩. Loss এবং Optimizer সেটআপ
    criterion = MSELoss()
    # লেয়ারের ওয়েইট এবং বায়াসগুলো অপটিমাইজারের কাছে পাঠানো হচ্ছে
    parameters = [layer.weights]
    if layer.bias is not None:
        parameters.append(layer.bias)
        
    optimizer = SGD(parameters, lr=0.01) # Learning rate 0.01

    print("🚀 Starting 10 Epochs of Training...\n")
    
    epochs = 500
    for epoch in range(epochs):
        # Step 1: Forward Pass (মডেলকে প্রেডিক্ট করতে বলা)
        predictions = layer(X)

        # Step 2: Compute Loss (মডেলের ভুল মাপা)
        loss = criterion(predictions, Y)

        # Step 3: Zero Gradients (আগের ভুলের মেমরি ক্লিয়ার করা)
        optimizer.zero_grad()

        # Step 4: Backward Pass (ভুলটা কোন দিকে হয়েছে সেটা মাপা)
        grad_out = criterion.backward()
        layer.backward(grad_out)

        # Step 5: Update Weights (C++ ড্রাইভার চালিত ওয়েইট আপডেট)
        optimizer.step()

        # প্রতিটা Epoch-এর লস প্রিন্ট করা
        print(f"🔄 Epoch [{epoch+1}/{epochs}] | Model Loss (Error): {loss.data:.4f}")

    print("\n✅ Training Complete!")
    print("------------------------------------------------")
    print(f"🎯 Target Values: \n{Y_data.flatten()}")
    print(f"🤖 Model Predictions (After Training): \n{layer(X).data.flatten()}")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    train_photonic_network()