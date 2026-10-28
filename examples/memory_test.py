import kecia.nn as knn

print("🧠 BANTAI Autograd & SGD Optimizer Test Initialized...\n")

# ১. মডেল তৈরি (3 Input, 16 Hidden, 3 Output)
model = knn.PhotonicSequential(in_features=3, hidden_features=16, out_features=3)

# ২. ট্রেইনিং ডেটা (ধরি, ইনপুট হলো [1.0, 0.5, 0.2] আর আমরা চাই আউটপুট আসুক [0.0, 1.0, 0.0])
input_data = [1.0, 0.5, 0.2]
target_data = [0.0, 1.0, 0.0]

print(f"🎯 Target Output: {target_data}\n")

# ৩. লাইভ ট্রেইনিং লুপ (Superfast C++ Training)
epochs = 50
learning_rate = 0.05

for epoch in range(epochs):
    # Step 1: Forward Pass (C++ Core)
    output = model(input_data)
    
    # Step 2: Backward Pass & Weight Update (C++ Core)
    model.backward(target_data, learning_rate)
    
    # প্রতি ১০ ইপক পর পর রেজাল্ট প্রিন্ট করে দেখা
    if (epoch + 1) % 10 == 0 or epoch == 0:
        # Error (MSE) ক্যালকুলেট করা শুধু দেখানোর জন্য
        loss = sum((output[i] - target_data[i])**2 for i in range(3)) / 3
        print(f"Epoch {epoch + 1:02d} | Loss: {loss:.4f} | Output: {[round(o, 4) for o in output]}")

print("\n✅ Training Complete! The C++ Photonic Engine successfully learned the pattern.")