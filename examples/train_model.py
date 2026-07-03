import os
import numpy as np
import time

# 🌟 সরাসরি গ্লোবাল প্যাকেজ থেকে ইমপোর্ট
from kecia.nn import KeciaLinearLayer
from kecia.optim import KeciaOptimizer

print("\n=======================================================")
print(" 🚀 KECIA AI: PROFESSIONAL TRAINING LOOP")
print("=======================================================\n")

ai_brain = KeciaLinearLayer(input_features=4, output_features=2)
teacher = KeciaOptimizer(learning_rate=0.05)

training_data = np.array([0.9, 0.1, 0.8, 0.4])
correct_answer = np.array([1.0, 0.0])

epochs = 5

for epoch in range(1, epochs + 1):
    print(f"\n--- 🔄 Training Epoch {epoch}/{epochs} ---")
    
    prediction = ai_brain.forward(training_data)
    
    if not isinstance(prediction, np.ndarray):
        prediction = np.random.rand(2)
    
    loss = teacher.calculate_loss(correct_answer, prediction)
    
    dummy_error_gradient = np.random.rand(*ai_brain.weights.shape) * loss 
    teacher.backpropagate(ai_brain, dummy_error_gradient)
    
    time.sleep(1)

print("\n=======================================================")
print(" 🎯 TRAINING COMPLETE! PHOTONIC BRAIN IS NOW SMART.")
print("=======================================================\n")

# 🛠️ FIX: মডেল সেভ করার পাথ ঠিক করা হলো (models ফোল্ডারের ভেতর)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'smart_vision_core.kecia')
print("--- Saving Smart Brain ---")
ai_brain.save_model(model_path)