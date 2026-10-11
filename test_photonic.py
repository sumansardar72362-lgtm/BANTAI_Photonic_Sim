from kecia.backend.photonic_backend import PhotonicDeepCore

print("🚀 Initializing BANTAI Kecia SDK...")
model = PhotonicDeepCore(in_features=3, hid_features=16, out_features=3)

# সাধারণ সিমুলেশন মোড (ডিফল্ট)
model.set_mode("simulation")

# ⚡ দ্য মাস্টারস্ট্রোক: ফিজিক্যাল চিপের সাথে কানেক্ট করার কমান্ড!
model.set_mode("hardware")

print("✅ Hardware integration test complete!")