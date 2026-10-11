import numpy as np
import json
from kecia.backend.photonic_backend import PhotonicDeepCore

print("🚀 BANTAI SDK: Deep Learning Training Initialized...")

# ১. কঠিন ডেটাসেট (যেখানে সিঙ্গেল লেয়ার কনফিউজড হতো)
dataset = [
    # 🎯 Vlog/Feedback (0)
    {"features": [0.0, 0.026, 0.129], "target": 0},  # Feedback 1
    {"features": [0.8, 0.05, 0.2], "target": 0},     # Feedback 2 (Clear Face)
    {"features": [0.0, 0.15, 0.4], "target": 0},     # Feedback 3 (No Face, Hand Motion)
    
    # 🎯 Gaming/Animation (1)
    {"features": [0.0, 0.03, 0.9], "target": 1},     # Gaming (No Face, High Graphics)
    {"features": [0.0, 0.01, 0.8], "target": 1},     # Gaming 2
    
    # 🎯 Dance/Action/Sports (2)
    {"features": [0.0, 0.294, 0.179], "target": 2},  # Boxing (No Face, High Motion)
    {"features": [0.0, 0.95, 0.4], "target": 2},     # Extreme Dance
]

# ডেটা প্রসেসিং (NumPy ম্যাট্রিক্স)
X = np.array([d["features"] for d in dataset])
Y = np.zeros((len(dataset), 3))
for i, d in enumerate(dataset):
    Y[i, d["target"]] = 1.0

# ২. নিউরাল নেটওয়ার্ক আর্কিটেকচার (৩ ইনপুট -> ১৬ হিডেন নিউরন -> ৩ আউটপুট)
in_features = 3
hidden_features = 16
out_features = 3

np.random.seed(42)
W1 = np.random.randn(in_features, hidden_features) * np.sqrt(2. / in_features)
b1 = np.zeros((1, hidden_features))
W2 = np.random.randn(hidden_features, out_features) * np.sqrt(2. / hidden_features)
b2 = np.zeros((1, out_features))

# ৩. Deep Learning Training Loop (Backpropagation)
epochs = 1000
learning_rate = 0.05

for epoch in range(epochs):
    # Forward Pass
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)  # ReLU
    
    # Error Calculation
    loss = np.mean(np.square(A2 - Y))
    
    # Backward Pass (Г্যাডিয়েন্ট ডিসেন্ট)
    dA2 = 2 * (A2 - Y) / len(X)
    dZ2 = dA2 * (Z2 > 0)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # Update Weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Deep Loss: {loss:.4f}")

# ৪. ⚡ ট্রেইন করা জ্ঞান তোমার C++ ফোটোনিক ইঞ্জিনে পুশ করা!
print("\n🧠 Injecting Deep Knowledge into BANTAI C++ Photonic Core...")
deep_core = PhotonicDeepCore(in_features, hidden_features, out_features)

deep_core.set_w1(W1.tolist())
deep_core.set_b1(b1[0].tolist())
deep_core.set_w2(W2.tolist())
deep_core.set_b2(b2[0].tolist())

# ৫. জ্ঞান ফাইলে সেভ করা
model_data = {
    "version": "2.0-Deep",
    "engine": "BANTAI Multi-Layer Photonic Core",
    "architecture": [3, 16, 3],
    "W1": W1.tolist(), "b1": b1[0].tolist(),
    "W2": W2.tolist(), "b2": b2[0].tolist()
}

with open("bantai_deep_model.json", "w") as f:
    json.dump(model_data, f, indent=4)

print("💾 Deep Learning Model saved to 'bantai_deep_model.json'!")