import cv2
import numpy as np
import time
import sys
import os

# src ফোল্ডার কানেক্ট করা
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.core.tensor5d import Tensor5D
# তোমার যদি PhotonicVisionModel অন্য ফাইলে থাকে, তবে সেখান থেকে ইম্পোর্ট করে নিও।
# আপাতত আমরা ধরে নিচ্ছি মডেল আর্কিটেকচারটা এখানেই ইম্পোর্ট করা যাবে।

print("🚀 Initializing BANTAI Photonic Vision Pipeline...")

# ১. মডেল লোড করা (তুমি যে .kecia ফাইল সেভ করেছ, সেখান থেকে লোড হবে)
# এখানে তোমার load_model ফাংশনটা ব্যবহার করবে, টেস্টের জন্য আমরা আপাতত একটা ডামি মডেল ফাংশন রাখছি
def predict_frame_with_photonic_engine(frame_pixels):
    """
    এখানে তোমার আসল PhotonicVisionModel-এর forward pass হবে।
    আপাতত ডেমোর জন্য আমরা পিক্সেলের ইনটেনসিটি দিয়ে স্প্যাম স্কোর বের করছি।
    """
    # C++ ইঞ্জিনে ডেটা পাঠানোর সিমুলেশন
    avg_intensity = np.mean(frame_pixels)
    spam_score = (avg_intensity / 255.0) * 10  # 0 থেকে 10 এর মধ্যে স্কোর
    return spam_score

# ২. OpenCV দিয়ে ওয়েবক্যাম চালু করা (0 মানে ডিফল্ট ক্যামেরা)
cap = cv2.VideoCapture("c:/Users/uday/Videos/dddddd.mp4")

if not cap.isOpened():
    print("❌ Error: Could not open Webcam / Video File.")
    exit()

print("🟢 Live Video Feed Started! Press 'q' to Quit.")

while True:
    start_time = time.time()
    
    # ক্যামেরা থেকে ফ্রেম পড়া
    ret, frame = cap.read()
    if not ret:
        break

    # ৩. এআই-এর জন্য ফ্রেম প্রসেসিং (Preprocessing)
    # তোমার মডেল 32x32 পিক্সেল সাপোর্ট করে, তাই ছবিটাকে ছোট করে নিচ্ছি
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # সাদাকালো করা
    resized_frame = cv2.resize(gray_frame, (32, 32))      # 32x32 পিক্সেলে কনভার্ট
    
    # ⚡ ৪. C++ ফোটোনিক ইঞ্জিনে ফ্রেম পাঠানো
    # এখানে তোমার Tensor5D তে কনভার্ট করে মডেলে পাঠাবে
    score = predict_frame_with_photonic_engine(resized_frame)
    
    # ৫. ডিসিশন মেকিং
    if score > 5.0:
        label = f"WARNING: SPAM/FAKE ({score:.2f})"
        color = (0, 0, 255)  # লাল কালার
    else:
        label = f"CLEAN/REAL ({score:.2f})"
        color = (0, 255, 0)  # সবুজ কালার

    # ৬. স্ক্রিনের ওপর লাইভ রেজাল্ট (HUD) প্রিন্ট করা
    fps = 1.0 / (time.time() - start_time)
    
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Engine: C++ Photonic Core | FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Processing Res: 32x32", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # ভিডিও দেখানো
    cv2.imshow("Protikriya Live AI Vision", frame)

    # 'q' চাপলে বন্ধ হয়ে যাবে
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# কাজ শেষে ক্যামেরা বন্ধ করা
cap.release()
cv2.destroyAllWindows()