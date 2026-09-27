import sys
import os

# src ফোল্ডারটিকে পাথ হিসেবে চিনিয়ে দেওয়া হচ্ছে
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kecia.nn.linear5d import Linear5D

print("🔥 BANTAI Kecia Neural Network Test Started!\n")

# একটি ডামি নিউরাল নেটওয়ার্ক লেয়ার তৈরি করা হলো (৩টা ইনপুট, ২টা আউটপুট)
layer = Linear5D(in_features=3, out_features=2)

# ডামি ইনপুট ডেটা (যেমন: ভিডিও ফ্রেমের পিক্সেল বা ফিচারের ইনটেনসিটি)
my_inputs = [1.5, 2.0, 3.0]

print(f"\n📥 Network Input: {my_inputs}")

# ফরোয়ার্ড পাস (এখানেই C++ ইঞ্জিন কাজ করবে)
network_output = layer.forward(my_inputs)

print(f"\n🚀 Final Network Output (Processed by C++): {network_output}")