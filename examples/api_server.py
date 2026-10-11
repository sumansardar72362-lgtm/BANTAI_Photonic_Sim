from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import os
import json
import uvicorn

# ⚡ তোমার BANTAI SDK-এর নতুন Deep Learning ইঞ্জিন
from kecia.backend.photonic_backend import PhotonicDeepCore

app = FastAPI(
    title="Protikriya API",
    description="Powered by BANTAI Photonic Deep Core (Multi-Layer)",
    version="2.0-Deep"
)

# 🧠 1. Deep Core ইনিশিয়ালাইজ করা (3 Input -> 16 Hidden -> 3 Output)
print("🧠 Initializing BANTAI Photonic Deep Core...")
photonic_brain = PhotonicDeepCore(in_features=3, hid_features=16, out_features=3)

# 🎯 2. ট্রেইন করা মডেল লোড করা
model_file = "bantai_deep_model.json"
if os.path.exists(model_file):
    with open(model_file, "r") as f:
        model_data = json.load(f)
        # Deep Learning-এর ৪টি লেয়ারের ওয়েট ও বায়াস সেট করা
        photonic_brain.set_w1(model_data["W1"])
        photonic_brain.set_b1(model_data["b1"])
        photonic_brain.set_w2(model_data["W2"])
        photonic_brain.set_b2(model_data["b2"])
    print(f"✅ Deep Knowledge permanently loaded from {model_file}!")
else:
    print("⚠️ Warning: No deep model found! Run train_bantai.py first.")

CATEGORIES = {
    0: {"name": "Vlog/Feedback", "action": "PUBLISH_ALLOWED"},
    1: {"name": "Gaming/Animation", "action": "BLOCK - GAMING"},
    2: {"name": "Dance/Action/Sports", "action": "BLOCK - ACTION"}
}

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except AttributeError:
    face_cascade = None

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "BANTAI Deep-Photonic v2.0"}

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(temp_video_path)
    
    frame_count = 0
    total_motion = 0
    total_complexity = 0
    face_detected_frames = 0
    
    ret, prev_frame = cap.read()
    if not ret:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return {"error": "Video cannot be read"}
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= 30: 
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Motion
        frame_diff = cv2.absdiff(prev_gray, gray_frame)
        total_motion += np.mean(frame_diff)
        
        # Complexity
        edges = cv2.Canny(gray_frame, 100, 200)
        total_complexity += (np.sum(edges > 0) / edges.size)
        
        # Face
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                face_detected_frames += 1
                
        prev_gray = gray_frame
        frame_count += 1

    cap.release()
    if os.path.exists(temp_video_path): os.remove(temp_video_path)

    if frame_count == 0:
        return {"error": "Video processing failed"}

    avg_motion = (total_motion / frame_count) / 100.0  
    avg_complexity = (total_complexity / frame_count) * 10.0
    face_ratio = face_detected_frames / frame_count

    # 🚀 Deep Learning ইঞ্জিনের ভেতরে ডেটা পাঠানো
    nn_input = [face_ratio, avg_motion, avg_complexity]
    nn_output = photonic_brain.forward(nn_input)

    # Argmax (যেই নিউরন সবচেয়ে বেশি ফায়ার করবে)
    best_match_index = int(np.argmax(nn_output)) if sum(nn_output) > 0 else 0
    decision = CATEGORIES[best_match_index]

    return {
        "filename": file.filename,
        "input_features": {
            "face_ratio": round(face_ratio, 3),
            "motion": round(avg_motion, 3),
            "complexity": round(avg_complexity, 3)
        },
        "deep_network_scores": [round(val, 4) for val in nn_output],
        "category": decision["name"],
        "action": decision["action"],
        "powered_by": "BANTAI Multi-Layer SDK"
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)