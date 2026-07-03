import numpy as np
# আমাদের আগের বানানো নিউরাল নেটওয়ার্ক ইমপোর্ট করছি
from kecia_nn import KeciaLinearLayer

print("\n=======================================================")
print(" 👁️ KECIA AI VISION: VIDEO FRAME PROCESSING TEST")
print("=======================================================\n")

# ধাপ ১: ভিডিও ফ্রেমের পিক্সেল ডেটা সিমুলেট করা
# ধরো, আমরা ২৮x২৮ পিক্সেলের একটি ভিডিও ফ্রেম নিচ্ছি (মোট ৭৮৪ পিক্সেল)
print("[Vision] Capturing Video Frame (28x28 resolution)...")
video_frame_pixels = np.random.rand(784) 

# ধাপ ২: এআই ভিশন লেয়ার তৈরি করা
# ৭৮৪টি পিক্সেল ইনপুট নেবে এবং আউটপুটে ৩টি জিনিস খুঁজবে: (১. মানুষ, ২. প্রোডাক্ট, ৩. ব্যাকগ্রাউন্ড)
vision_layer = KeciaLinearLayer(input_features=784, output_features=3)

# ধাপ ৩: পুরো ফ্রেমের ডেটা ফোটোনিক চিপে পাঠানো
print("\n[Vision] Sending 784 Pixels to Photonic Chip for Object Detection...")
detection_result = vision_layer.forward(video_frame_pixels)

print("\n=======================================================")
print(" ✅ FRAME PROCESSED SUCCESSFULLY BY KECIA CHIP!")
print("=======================================================\n")