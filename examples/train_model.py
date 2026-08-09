import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kecia.core.tensor5d import Tensor5D
from src.kecia.nn import Linear5D, ReLU5D, Sequential, MSELoss
from src.kecia.optim.sgd import SGD

def train_photonic_network():
    print("\n🧠 --- BANTAI AI DEEP TRAINING (CLEAN API) --- 🧠\n")

    X = Tensor5D(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
    Y = Tensor5D(np.array([[2.0], [4.0], [6.0]], dtype=np.float32))

    print("🛠️ Building Network with Sequential API...")
    # 💥 ম্যাজিক: পুরো মডেল মাত্র ৪ লাইনে!
    model = Sequential(
        Linear5D(in_features=1, out_features=8),
        ReLU5D(),
        Linear5D(in_features=8, out_features=1)
    )

    criterion = MSELoss()
    
    # 💥 ম্যাজিক: আর কোনো ম্যানুয়াল লুপ নেই!
    optimizer = SGD(model.get_parameters(), lr=0.005)

    print("\n🚀 Starting 500 Epochs of Deep Training...\n")
    epochs = 500
    loss_history = []

    for epoch in range(epochs):
        # 💥 ম্যাজিক: ফরোয়ার্ড এবং ব্যাকওয়ার্ড পাস এখন একদম সিম্পল!
        predictions = model(X)
        loss = criterion(predictions, Y)
        
        loss_history.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))

        optimizer.zero_grad()
        model.backward(criterion.backward())
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"🔄 Epoch [{epoch+1}/{epochs}] | Model Loss: {loss.data:.4f}")

    print("\n✅ Deep Training Complete!")
    print(f"🤖 Model Predictions: \n{model(X).data.flatten()}")
    
    print("\n📈 Generating Loss Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), loss_history, color='purple', linewidth=2, label='Training Loss')
    plt.title('BANTAI Kecia - Clean API Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Error)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_photonic_network()