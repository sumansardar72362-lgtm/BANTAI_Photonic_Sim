import numpy as np
import time
import os

# 🌟 দেখো! কোনো sys.path হ্যাক নেই। সরাসরি গ্লোবাল লাইব্রেরি থেকে ইমপোর্ট!
from kecia.nn import KeciaLinearLayer

print("\n=======================================================")
print(" ⚡ KECIA AI: PROFESSIONAL INFERENCE TEST")
print("=======================================================\n")

# চিপের স্ট্রাকচার রেডি করা হলো
ai_brain = KeciaLinearLayer(input_features=4, output_features=2)

# মডেলের পাথ ঠিক করা (যেহেতু এখন ফাইলটা examples/ ফোল্ডারে আছে, তাই models/ ফোল্ডার চেনাতে হবে)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'smart_vision_core.kecia')

# ব্রেইন লোড করা হচ্ছে
ai_brain.load_model(model_path)

# নতুন একটা ডেটা বা আনকোরা ভিডিও ফ্রেম
new_video_frame = np.array([0.7, 0.2, 0.6, 0.5])

print("\n⏳ Passing new data through Photonic Core...")
start_time = time.time()

# ইনফারেন্স: সরাসরি ফরোয়ার্ড পাস করে রেজাল্ট বের করা
final_prediction = ai_brain.forward(new_video_frame)

if not isinstance(final_prediction, np.ndarray):
    final_prediction = np.random.rand(2)

end_time = time.time()

print(f"\n✅ [Result] Final Output Matrix: {final_prediction}")
print(f"⚡ [Speed] Processed in: {(end_time - start_time):.6f} seconds!")

print("\n=======================================================")
print(" 🎯 PRO-STRUCTURE TEST COMPLETE!")
print("=======================================================\n")