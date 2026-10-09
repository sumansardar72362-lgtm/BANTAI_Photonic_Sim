import sys
import os
import numpy as np

# 🔗 পাইথনকে বলে দেওয়া হচ্ছে যে মেইন ইঞ্জিনটা src ফোল্ডারে আছে
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D
from kecia.nn.conv2d import Conv2D
from kecia.nn.linear5d import Linear5D
from kecia.nn.activation import ReLU5D
from kecia.optim import Adam
from kecia.io import save_model  # 👈 নতুন যোগ করা হলো

# 🛠️ ছোট্ট একটি Flatten ক্লাস
class Flatten:
    def forward(self, x):
        self.last_shape = x.shape
        out = x.data.reshape(x.shape[0], -1)
        return Tensor5D(out, device=x.device)
        
    def backward(self, grad_out):
        grad_data = grad_out.data if isinstance(grad_out, Tensor5D) else grad_out
        return Tensor5D(grad_data.reshape(self.last_shape), device='photonic')
        
    def parameters(self): return []
    def __call__(self, x): return self.forward(x)

# 🧠 Photonic Vision Architecture
class PhotonicVisionModel:
    def __init__(self):
        self.layers = [
            Conv2D(in_channels=1, out_channels=4, kernel_size=3),
            ReLU5D(threshold=0.01),
            Flatten(),
            Linear5D(4 * 6 * 6, 16),
            ReLU5D(),
            Linear5D(16, 5)
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad
        
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

# ==========================================
# 🚀 ট্রেইনিং সেটআপ
# ==========================================

print("Preparing and Normalizing Data...")
X_train = np.random.rand(10, 1, 8, 8).astype(np.float32) 
X_train = X_train / np.max(np.abs(X_train)) 

Y_train = np.zeros((10, 5), dtype=np.float32)
for i in range(10): Y_train[i, np.random.randint(0, 5)] = 1.0

model = PhotonicVisionModel()
optim = Adam(model.parameters(), lr=0.005) 

print("\n🚀 Starting Photonic Vision Training Loop...")
for epoch in range(20):
    # Forward Pass
    preds = model.forward(Tensor5D(X_train))
    
    # Loss (Mean Squared Error)
    loss = np.mean((preds.data - Y_train) ** 2)
    
    # Backward Pass
    grad = 2.0 * (preds.data - Y_train) / 10.0
    model.backward(Tensor5D(grad))
    
    # Weights Update
    optim.step()
    
    print(f"Epoch {epoch+1:02d}/20 | Loss: {loss:.6f}")

# ==========================================
# 💾 ট্রেইনিং শেষে মডেল সেভ করা (5D Glass Memory)
# ==========================================
print("\nInitiating 5D Glass Memory Storage...")
save_model(model.parameters(), "protikriya_vision_v1.kecia") # 👈 মডেল সেভ হচ্ছে