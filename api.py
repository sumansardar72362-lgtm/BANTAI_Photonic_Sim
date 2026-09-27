from fastapi import FastAPI
from pydantic import BaseModel
# তোমার লাইব্রেরি ইমপোর্ট করা হলো
# from kecia import Sequential, Dense 

# API সার্ভার তৈরি
app = FastAPI(title="BANTAI Photonic API", version="1.0")

# ইউজার বা ওয়েবসাইট থেকে কী ধরনের ডেটা আসবে, তার স্ট্রাকচার
class VideoData(BaseModel):
    video_id: str
    features: list[float]  # ভিডিওর ফিচার ডেটা (যেমন: লেন্থ, পিক্সেল ডেনসিটি ইত্যাদি)

# এপিআইয়ের হোমপেজ
@app.get("/")
def home():
    return {"message": "Welcome to BANTAI Photonic AI Engine", "status": "Active"}

# প্রেডিকশন নেওয়ার জন্য POST রাউট (যেখানে PHP ব্যাকএন্ড থেকে ডেটা আসবে)
@app.post("/predict/video_spam")
def predict_video_spam(data: VideoData):
    # (ভবিষ্যতে এখানে model.load("my_model.kecia") ব্যবহার করে রিয়েল মডেল লোড হবে)
    
    # আপাতত সিমুলেশন লজিক:
    feature_count = len(data.features)
    
    if feature_count > 0:
        # তোমার ফোটোনিক ইঞ্জিনে ডেটা পাঠানো হচ্ছে...
        spam_probability = 0.12 # (ধরে নিচ্ছি মডেল ১২% স্প্যাম পেয়েছে)
        status = "Clean" if spam_probability < 0.5 else "Spam Detected"
    else:
        spam_probability = 0.0
        status = "Invalid Data"

    # ওয়েবসাইটকে JSON ফরম্যাটে রেজাল্ট ফেরত দেওয়া
    return {
        "video_id": data.video_id,
        "spam_probability": f"{spam_probability * 100}%",
        "hardware_used": "Photonic VMM Simulator",
        "status": status
    }