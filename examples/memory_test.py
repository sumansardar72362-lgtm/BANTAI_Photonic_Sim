import os
import sys

# সরাসরি গ্লোবাল প্যাকেজ থেকে ইমপোর্ট
from kecia.nn import KeciaLinearLayer

print("\n=======================================================")
print(" 🧠 KECIA MEMORY MANAGEMENT TEST (PRO STRUCTURE)")
print("=======================================================\n")

# চিপ চালু করা হলো
ai_brain = KeciaLinearLayer(4, 2)

# 🛠️ FIX: মডেল সেভ করার পাথ ঠিক করা হলো (models ফোল্ডারের ভেতর)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vision_core.kecia')

# ১. চিপের বর্তমান মেমরি (Weights) সেভ করা হচ্ছে
print("\n--- Saving Current AI State ---")
ai_brain.save_model(model_path)

# ২. চিপের মেমরি ম্যানুয়ালি ডিলিট করে দেওয়া হচ্ছে (যাতে সে সব ভুলে যায়)
print("\n--- Erasing Memory (Simulating Power Off) ---")
ai_brain.weights = None 
print("⚠️ AI Brain is now Empty!")

# ৩. ফাইল থেকে মেমরি আবার লোড করা হচ্ছে 
print("\n--- Reloading AI State ---")
ai_brain.load_model(model_path)

print("\n=======================================================")
print(" 🎯 MEMORY TEST COMPLETE!")
print("=======================================================\n")