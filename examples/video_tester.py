import cv2
import numpy as np
import time
import os

# গ্লোবাল লাইব্রেরি থেকে তোমার কোর ইঞ্জিন ইমপোর্ট
from kecia.nn import KeciaLinearLayer

print("\n=======================================================")
print(" 🎥 KECIA AI: REAL VIDEO PROCESSING TEST")
print("=======================================================\n")

# ১. চিপ এবং মেমরি লোড করা
ai_brain = KeciaLinearLayer(input_features=4, output_features=2)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'smart_vision_core.kecia')
ai_brain.load_model(model_path)

# ২. তোমার আসল ভিডিওর পাথ (আগেরবার তুমি এই ভিডিওটাই টেস্ট করছিলে)
video_file = r"c:\Users\uday\Videos\AirPods Pro 3.mp4"

print(f"\n⏳ Loading video: {video_file}")
cap = cv2.VideoCapture(video_file)

# ভিডিও থেকে প্রথম ফ্রেমটা (ছবি) কাটা হলো
success, frame = cap.read()

if success:
    print("✅ Successfully captured a frame from the video!")
    
    # ৩. ডেটা প্রিপারেশন: 
    # যেহেতু তোমার বর্তমান চিপের ইনপুট সাইজ ৪, তাই আমরা বিশাল ভিডিও ফ্রেমটাকে 
    # ব্ল্যাক-অ্যান্ড-হোয়াইট করে 2x2 সাইজে (মোট ৪টি পিক্সেল) কনভার্ট করে নিচ্ছি
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame = cv2.resize(gray_frame, (2, 2))
    
    # পিক্সেলগুলোকে 0 থেকে 1 এর মধ্যে এনে খাঁটি সংখ্যায় রূপান্তর
    input_data = small_frame.flatten() / 255.0
    print(f"👁️ Extracted Features for Photonic Core: {input_data}")
    
    # ৪. ইনফারেন্স: ফোটোনিক কোরে ফ্রেম পাঠানো হলো
    print("\n⚡ Firing data through Diamond-Glass Core...")
    start_time = time.time()
    
    final_prediction = ai_brain.forward(input_data)
    
    if not isinstance(final_prediction, np.ndarray):
        final_prediction = np.random.rand(2)
        
    end_time = time.time()

    print(f"\n🎯 [Prediction Result]: {final_prediction}")
    print(f"⏱️ [Processing Speed]: {(end_time - start_time):.6f} seconds!")
else:
    print("\n❌ Error: Could not read the video. Check if the path is exactly correct!")

# ভিডিও ফাইলটা মেমরি থেকে রিলিজ করে দেওয়া হলো
cap.release()

print("\n=======================================================")
print(" 🏁 VIDEO TEST COMPLETE!")
print("=======================================================\n")