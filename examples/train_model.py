import sys
import os
import numpy as np

# মেইন প্রজেক্টের পাথ সেটআপ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kecia.core.tensor5d import Tensor5D
from src.kecia.nn.linear5d import Linear5D
from src.kecia.nn.activation import ReLU5D, Sigmoid5D  # 💥 নতুন অ্যাক্টিভেশন ইমপোর্ট
from src.kecia.nn.loss import MSELoss
from src.kecia.optim.sgd import SGD

def train_photonic_network():
    print("\n🧠 --- BANTAI AI DEEP TRAINING (WITH ACTIVATIONS) --- 🧠\n")

    # ১. ডেটাসেট (Task: ইনপুটকে দ্বিগুণ করা)
    X_data = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    Y_data = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)
    
    X = Tensor5D(X_data)
    Y = Tensor5D(Y_data)

    # ২. ডিপ মডেল ডিজাইন (1 Input -> 8 Hidden Neurons -> 1 Output)
    print("🛠️ Building 2-Layer Network with ReLU Activation...")
    layer1 = Linear5D(in_features=1, out_features=8)
    relu = ReLU5D()
    layer2 = Linear5D(in_features=8, out_features=1)

    # ৩. Loss এবং Optimizer
    criterion = MSELoss()
    
    # ডাইনামিক্যালি দুই লেয়ারের ওয়েইট এবং বায়াস কালেক্ট করা
    parameters = []
    for layer in [layer1, layer2]:
        parameters.append(layer.weights)
        if layer.bias is not None:
            parameters.append(layer.bias)
            
    # লার্নিং রেট একটু কমিয়ে দেওয়া হলো, কারণ লেয়ার বেশি
    optimizer = SGD(parameters, lr=0.005)

    print("\n🚀 Starting 500 Epochs of Deep Training...\n")
    
    epochs = 500
    for epoch in range(epochs):
        # ----------------------------------------------------
        # ⚡ Step 1: Forward Pass (ম্যাজিক এখানেই!)
        # ----------------------------------------------------
        h1 = layer1(X)          # ইনপুট গেল প্রথম লেয়ারে
        a1 = relu(h1)           # সিগন্যাল ফিল্টার হলো ফোটোনিক ReLU দিয়ে
        predictions = layer2(a1) # ফিল্টার করা সিগন্যাল গেল আউটপুট লেয়ারে

        # Step 2: Compute Loss
        loss = criterion(predictions, Y)

        # Step 3: Zero Gradients
        optimizer.zero_grad()

        # ----------------------------------------------------
        # 🔄 Step 4: Backward Pass (Chain Rule / ব্যাকপ্রপাগেশন)
        # ----------------------------------------------------
        grad_out = criterion.backward()         # লস থেকে প্রথম ভুল বের হলো
        grad_a1 = layer2.backward(grad_out)     # আউটপুট লেয়ার তার ভুল বুঝল
        grad_h1 = relu.backward(grad_a1)        # ReLU তার ভেতর দিয়ে ভুলটা পেছনের দিকে পাঠাল
        layer1.backward(grad_h1)                # প্রথম হিডেন লেয়ার তার ভুল বুঝল

        # Step 5: Update Weights (C++ ড্রাইভার)
        optimizer.step()

        # টার্মিনাল ক্লিন রাখার জন্য প্রতি ১০০ ইপকে রেজাল্ট প্রিন্ট করা
        if (epoch + 1) % 100 == 0:
            print(f"🔄 Epoch [{epoch+1}/{epochs}] | Model Loss (Error): {loss.data:.4f}")

    print("\n✅ Deep Training Complete!")
    print("------------------------------------------------")
    print(f"🎯 Target Values: \n{Y_data.flatten()}")
    
    # ফাইনাল প্রেডিকশনের জন্য আরেকবার ফরোয়ার্ড পাস
    h1_final = layer1(X)
    a1_final = relu(h1_final)
    final_preds = layer2(a1_final)
    
    print(f"🤖 Model Predictions (With ReLU): \n{final_preds.data.flatten()}")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    train_photonic_network()