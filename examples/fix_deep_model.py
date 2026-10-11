import json
import numpy as np

print("🛠️ Generating Perfect Deep Learning Matrix for BANTAI SDK...")

W1 = np.zeros((3, 16))
W1[0, 0] = 1.0  # Face
W1[1, 1] = 1.0  # Motion
W1[2, 2] = 1.0  # Complexity
b1 = np.zeros(16)

W2 = np.zeros((16, 3))

# Output 0: Vlog (Face থাকলে পয়েন্ট বাড়বে, কিন্তু Complexity-র জন্য কোনো পয়েন্ট কাটা হবে না!)
W2[0, 0] = 10.0
W2[1, 0] = -5.0  
W2[2, 0] = 0.0   # 👈 ম্যাজিক! ব্যাকগ্রাউন্ডে যাই থাকুক, Vlog-এর পয়েন্ট আর কাটা যাবে না।

# Output 1: Gaming 
W2[2, 1] = 15.0

# Output 2: Action (Motion-এর পাওয়ার আরও বাড়ানো হলো)
W2[1, 2] = 25.0

# Biases: Gaming-এর লিমিট অনেক বাড়িয়ে দেওয়া হলো (-8.0) 
# যাতে সাধারণ ব্যাকগ্রাউন্ড গেম হিসেবে ডিটেক্ট না হয়! শুধুমাত্র ফুল স্ক্রিন গ্রাফিক্স হলেই এটা ফায়ার করবে।
b2 = np.array([2.5, -8.0, -4.0]) 

model_data = {
    "version": "2.0-Deep",
    "engine": "BANTAI Multi-Layer Photonic Core",
    "W1": W1.tolist(),
    "b1": b1.tolist(),
    "W2": W2.tolist(),
    "b2": b2.tolist()
}

with open("bantai_deep_model.json", "w") as f:
    json.dump(model_data, f, indent=4)

print("✅ Balanced Multi-Layer Weights Injected into 'bantai_deep_model.json'!")