import os
from dotenv import load_dotenv

# .env ফাইল থেকে ডেটা লোড করা
load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY is missing! Please check your .env file.")

# রিপোর্টের ফোল্ডার পাথ
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
REPORT_FILE_NAME = "Development_Journal.docx"
REPORT_FILE_PATH = os.path.join(REPORTS_DIR, REPORT_FILE_NAME)

# ফোল্ডার না থাকলে তৈরি করে নেবে
os.makedirs(REPORTS_DIR, exist_ok=True)