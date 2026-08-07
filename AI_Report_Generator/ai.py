from google import genai
from config import GEMINI_API_KEY

def generate_report(date, message, diff):
    if not message or not diff:
        return "Error: Not enough data to generate a report."
    
    # Gemini ক্লায়েন্ট ইনিশিয়ালাইজ করা
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # AI-কে দেওয়া নির্দেশ (Prompt)
    prompt = f"""
    You are an expert CTO and technical writer. I am giving you a git commit message and the exact code changes (git diff).
    Please analyze them and write a concise, professional daily development journal entry.
    This report is extremely confidential and will be used for internal tracking, future pitch decks, and investor updates. Make it sound professional and highlight the value of the technical progress.
    
    Date: {date}
    Commit Message: {message}
    
    Code Diff (limited to 3000 chars to avoid token overload):
    {diff[:3000]}
    
    Format the output EXACTLY like this (do not use markdown bolding like ** , just use plain text and bullet points):
    
    Completed
    * [Task 1 - explain what it does]
    * [Task 2 - explain its value]
    
    Summary
    [1-2 sentences summarizing the technical progress and its business/project value]
    
    Tomorrow
    [Suggest 1 logical next step based on today's current changes]
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"