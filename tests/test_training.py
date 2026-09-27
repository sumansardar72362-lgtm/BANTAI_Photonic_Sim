import sys
import os
import numpy as np

# src ফোল্ডার কানেক্ট করা
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D
from kecia.nn import Linear5D, ReLU5D, Sequential, MSELoss
from kecia.optim.adam import Adam
# 💥 তোমার নতুন DataLoader ইমপোর্ট করা হলো!
from kecia.data.dataloader import DataLoader

def train_with_dataloader():
    print("\n📦 --- BANTAI AI: PHOTONIC BATCH TRAINING --- 📦\n")

    # ১. একটু বড় ডামি ডেটাসেট তৈরি করা (20 Samples)
    # ইনপুট: ১, ২, ৩... ২০ | আউটপুট হবে দ্বিগুণ: ২, ৪, ৬... ৪০
    raw_X = np.array([[float(i)] for i in range(1, 21)], dtype=np.float32)
    raw_Y = np.array([[float(i * 2)] for i in range(1, 21)], dtype=np.float32)

    # ২. DataLoader সেটআপ (Batch Size = 4)
    # ২০টা ডেটা ৪টা করে ভাগ হয়ে মোট ৫টা ব্যাচ তৈরি করবে
    dataloader = DataLoader(raw_X, raw_Y, batch_size=4, shuffle=True)

    print("\n🛠️ Building Photonic Network...")
    model = Sequential(
        Linear5D(in_features=1, out_features=8),
        ReLU5D(),
        Linear5D(in_features=8, out_features=1)
    )

    criterion = MSELoss()
    optimizer = Adam(model.get_parameters(), lr=0.01)

    print("\n🚀 Starting Training Loop with Batch Processing...\n")
    epochs = 500

    for epoch in range(epochs):
        epoch_loss = 0.0

        # 💥 ম্যাজিক: ইনার লুপে ব্যাচ প্রসেসিং! 
        # dataloader নিজে থেকেই 4টি করে ডেটা Tensor5D বানিয়ে লুপে পাঠাবে
        for batch_X, batch_Y in dataloader:
            
            # 1. Forward Pass (C++ কোর দিয়ে প্রসেস হবে)
            predictions = model(batch_X)
            
            # 2. Error / Loss Calculation
            loss = criterion(predictions, batch_Y)
            
            # Loss ট্র্যাকিং
            batch_loss = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            epoch_loss += batch_loss

            # 3. Backward Pass & Optimize (ভুল শুধরানো)
            optimizer.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()

        # প্রতি এপোচের শেষে ৫টি ব্যাচের গড় (Average) লস বের করা
        avg_epoch_loss = epoch_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"🔄 Epoch [{epoch+1:03d}/{epochs}] | Average Batch Loss: {avg_epoch_loss:.4f}")

    print("-" * 50)
    print("\n✅ Batch Training Complete!")
    
    # নতুন একটি অদেখা (Unseen) ডেটা দিয়ে টেস্ট করা
    test_input = 25.0
    test_data = Tensor5D(np.array([[test_input]], dtype=np.float32))
    predicted_output = model(test_data).data.flatten()[0]
    
    print(f"🤖 AI Prediction for Input {test_input} (Expected ~50.0): {predicted_output:.2f}")

if __name__ == "__main__":
    train_with_dataloader()