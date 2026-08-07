import os

# ১. ঠিক এইখানে তোমার সম্পূর্ণ API Key-টি ডাবল কোটেশনের (" ") ভেতরে বসিয়ে দাও
GEMINI_API_KEY = "AQ.Ab8RN6IEA6iVDoBNxjFqr998pVPCtXhp77OKty_GQotyeoLNNQ"

# রিপোর্টের ফোল্ডার পাথ
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
REPORT_FILE_NAME = "Development_Journal.docx"
REPORT_FILE_PATH = os.path.join(REPORTS_DIR, REPORT_FILE_NAME)

# ফোল্ডার না থাকলে তৈরি করে নেবে
os.makedirs(REPORTS_DIR, exist_ok=True)