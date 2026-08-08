def generate_report(date, message, diff):
    if not message:
        return "Error: No commit message found."
    
    # এখানে কোনো AI নেই, এটি শুধু তোমার কাজটাকে প্রফেশনালভাবে সাজিয়ে দেবে
    report = f"""Task Completed:
- {message.capitalize()}

Status: Done
Security Note: All code changes are securely tracked in the local Git repository.
"""
    return report.strip()