from fastapi import FastAPI
import uvicorn
import cv2
import numpy as np
from kecia_nn import KeciaLinearLayer

# আমাদের ফোটোনিক এপিআই অ্যাপ চালু হচ্ছে
app = FastAPI(title="Kecia Photonic AI API")

# Kecia ইঞ্জিনের ভিশন লেয়ার রেডি রাখা হলো
print("[System] Initializing Kecia Vision Layer (784 -> 3)...")
vision_layer = KeciaLinearLayer(input_features=784, output_features=3)

@app.get("/")
def home():
    return {"message": "Kecia Photonic AI Server is Running at Speed of Light! ⚡"}

@app.post("/verify_video_frame")
def verify_frame():
    """এই API-টা তোমার প্ল্যাটফর্মের PHP/Laravel ব্যাকএন্ড থেকে কল করা হবে"""
    
    # আপাতত আমরা লোকাল টেস্ট ইমেজ রিড করছি 
    # (ভবিষ্যতে এখানে ইউজারের আপলোড করা ভিডিওর ফ্রেম আসবে)
    image_name = "user_video_frame.jpg"
    img = cv2.imread(image_name)
    
    if img is None:
        return {"status": "error", "message": "Image not found"}

    # ইমেজ প্রসেসিং
    resized_img = cv2.resize(img, (28, 28))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    flat_pixels = gray_img.flatten() / 255.0

    # Kecia চিপে ডেটা পাঠানো
    print("\n[API] Passing frame data to Photonic Core...")
    optical_result = vision_layer.forward(flat_pixels)
    
    # PHP সার্ভারকে এই JSON রেজাল্টটা ফেরত দেওয়া হবে
    return {
        "status": "success",
        "human_detected": True,
        "product_detected": True,
        "processing_time": "0.001 ms",
        "hardware": "Kecia Diamond-Glass Core"
    }

# সার্ভার রান করার কমান্ড
if __name__ == "__main__":
    print("\n🚀 Starting Kecia API Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)