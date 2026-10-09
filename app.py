from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(title="Protikriya AI Engine")

# PHP থেকে আসা ডেটার স্ট্রাকচার
class VideoData(BaseModel):
    video_id: str
    user_id: str
    video_url: str

@app.get("/")
def read_root():
    return {"message": "BANTAI Photonic AI Core is Active! ⚡"}

@app.post("/check_video")
def check_video_content(data: VideoData):
    print(f"\n📥 PHP Backend থেকে রিকোয়েস্ট এসেছে!")
    print(f"ভিডিও ID: {data.video_id}, User ID: {data.user_id}")
    
    # এখানে আমাদের C++ AI ইঞ্জিন কাজ করবে (আপাতত ডামি স্প্যাম স্কোর দিচ্ছি)
    # ভবিষ্যতে এখানে তোমার PhotonicVisionModel বসবে
    spam_score = random.uniform(0.1, 9.9)
    is_fake = spam_score > 7.0  # ৭ এর বেশি হলে ফেক ধরব
    
    return {
        "status": "success",
        "video_id": data.video_id,
        "spam_score": round(spam_score, 2),
        "decision": "FAKE/SPAM" if is_fake else "CLEAN/REAL",
        "action_required": "DELETE_VIDEO" if is_fake else "PUBLISH_VIDEO"
    }