import subprocess
import sys

print("⏳ Downloading and Installing OpenCV... Please wait a few seconds.")
# এই কোডটি টার্মিনাল ছাড়াই সরাসরি তোমার VS Code-এর পাইথন দিয়ে OpenCV ডাউনলোড করবে
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
print("✅ OpenCV installed successfully! You are ready to go.")