from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI(
    title="Protikriya Smart Video Moderation API",
    description="BANTAI Photonic Core - Multi-Category Action Recognition",
    version="2.0.0"
)

print("🚀 Starting BANTAI Smart AI Engine (Multi-Category)...")

# 🛠️ Safe Face Detection Initialization (ক্র্যাশ-প্রুফ লজিক)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except AttributeError:
    print("⚠️ Warning: OpenCV Face module not found. Running in 'Motion & Complexity' mode only.")
    face_cascade = None

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "BANTAI-Kecia v2.0", "message": "Smart Engine Ready!"}

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    cap = cv2.VideoCapture(temp_file_path)
    
    frame_count = 0
    analyzed_frames = 0
    
    total_faces_detected = 0
    total_motion_score = 0
    total_color_complexity = 0
    
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # প্রতি ৩০ নম্বর ফ্রেম স্ক্যান করব
        if frame_count % 30 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ১. ফেস ডিটেকশন (যদি OpenCV সাপোর্ট করে)
            if face_cascade is not None and hasattr(face_cascade, 'detectMultiScale'):
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    total_faces_detected += 1
            
            # ২. কালার ও এজ (Edge) ডিটেকশন (Gaming ভিডিও চেনার জন্য)
            edges = cv2.Canny(gray_frame, 100, 200)
            edge_score = np.mean(edges) / 255.0
            total_color_complexity += edge_score
            
            # ৩. মোশন বা ড্যান্স ডিটেকশন
            if prev_gray is not None:
                frame_diff = cv2.absdiff(prev_gray, gray_frame)
                motion = np.mean(frame_diff)
                total_motion_score += motion
            
            prev_gray = gray_frame
            analyzed_frames += 1
            
        frame_count += 1

    cap.release()
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        
    if analyzed_frames == 0:
        return {"error": "Video too short or unreadable."}
        
    avg_faces = total_faces_detected / analyzed_frames
    avg_motion = total_motion_score / analyzed_frames if total_motion_score else 0
    avg_complexity = total_color_complexity / analyzed_frames
    
    # ⚡ BANTAI ডিসিশন মেকিং লজিক
    detected_category = "Unknown"
    action = "REVIEW"
    
    if avg_faces >= 0.2:
        detected_category = "Feedback/Vlog"
        action = "PUBLISH_ALLOWED"
    elif avg_complexity > 0.15 and avg_faces < 0.1:
        detected_category = "Gaming/Animation"
        action = "BLOCK - GAMING NOT ALLOWED"
    elif avg_motion > 15.0:
        detected_category = "Dance/Action"
        action = "BLOCK - DANCE NOT ALLOWED"
    else:
        detected_category = "General/Safe"
        action = "PUBLISH_ALLOWED"

    return {
        "filename": file.filename,
        "detected_category": detected_category,
        "action_recommendation": action,
        "metrics": {
            "face_presence_ratio": round(avg_faces, 2),
            "motion_intensity": round(avg_motion, 2),
            "visual_complexity": round(avg_complexity, 2)
        },
        "engine_used": "BANTAI Multi-Category Vision"
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)