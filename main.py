from git_reader import get_latest_commit_data
from ai import generate_report
from doc_writer import save_to_word
from config import REPORT_FILE_PATH

def main():
    print("\n🔍 Reading Git repository...")
    date, message, diff = get_latest_commit_data()
    
    if diff is None:
        print(f"❌ {message}")
        return
        
    print(f"📝 Found commit on {date}: '{message}'")
    print("🤖 Analyzing code changes with Gemini AI...")
    
    report = generate_report(date, message, diff)
    
    if report.startswith("Error"):
        print(f"❌ {report}")
        return
        
    print("💾 Saving professional report to Word document...")
    save_to_word(date, report)
    
    print(f"🎉 Done! Your daily development journal has been updated.")
    print(f"📁 Check your report at: {REPORT_FILE_PATH}\n")

if __name__ == "__main__":
    main()