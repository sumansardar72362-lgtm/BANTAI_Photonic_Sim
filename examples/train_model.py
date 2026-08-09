import sys
import os
import numpy as np
import matplotlib.pyplot as plt  # 💥 নতুন গ্রাফ লাইব্রেরি যুক্ত হলো

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kecia.core.tensor5d import Tensor5D
from src.kecia.nn.linear5d import Linear5D
from src.kecia.nn.activation import ReLU5D
from src.kecia.nn.loss import MSELoss
from src.kecia.optim.sgd import SGD

def train_photonic_network():
    print("\n🧠 --- BANTAI AI DEEP TRAINING (WITH VISUALIZATION) --- 🧠\n")

    X_data = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    Y_data = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)
    
    X = Tensor5D(X_data)
    Y = Tensor5D(Y_data)

    print("🛠️ Building 2-Layer Network with ReLU Activation...")
    layer1 = Linear5D(in_features=1, out_features=8)
    relu = ReLU5D()
    layer2 = Linear5D(in_features=8, out_features=1)

    criterion = MSELoss()
    
    parameters = []
    for layer in [layer1, layer2]:
        parameters.append(layer.weights)
        if layer.bias is not None:
            parameters.append(layer.bias)
            
    optimizer = SGD(parameters, lr=0.005)

    print("\n🚀 Starting 500 Epochs of Deep Training...\n")
    
    epochs = 500
    loss_history = []  # 💥 লস (Error) সেভ করে রাখার জন্য খালি লিস্ট

    for epoch in range(epochs):
        # Forward Pass
        h1 = layer1(X)
        a1 = relu(h1)
        predictions = layer2(a1)

        # Compute Loss
        loss = criterion(predictions, Y)
        
        # 💥 গ্রাফ আঁকার জন্য প্রতি ইপকের লস সেভ করে রাখা হচ্ছে
        loss_history.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))

        # Backward Pass & Optimize
        optimizer.zero_grad()
        grad_out = criterion.backward()
        grad_a1 = layer2.backward(grad_out)
        grad_h1 = relu.backward(grad_a1)
        layer1.backward(grad_h1)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"🔄 Epoch [{epoch+1}/{epochs}] | Model Loss (Error): {loss.data:.4f}")

    print("\n✅ Deep Training Complete!")
    print(f"🤖 Model Predictions: \n{layer2(relu(layer1(X))).data.flatten()}")
    
    # ----------------------------------------------------
    # 📊 ম্যাজিক: ট্রেনিং শেষে গ্রাফ স্ক্রিনে দেখানো
    # ----------------------------------------------------
    print("\n📈 Generating Loss Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), loss_history, color='blue', linewidth=2, label='Training Loss')
    plt.title('BANTAI Kecia - AI Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs (Practice times)', fontsize=12)
    plt.ylabel('Loss (Error)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()  # এই লাইনের কারণে স্ক্রিনে গ্রাফ ভেসে উঠবে!

if __name__ == "__main__":
    train_photonic_network()