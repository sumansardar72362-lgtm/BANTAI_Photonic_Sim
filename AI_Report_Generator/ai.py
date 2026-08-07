from gpt4all import GPT4All

def generate_report(date, message, diff):
    if not message:
        return "Error: Not enough data."
    
    print("\n🧠 Loading local offline AI model...")
    model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    
    # এআই-কে কোডের হিজিবিজি থেকে দূরে রেখে শুধু আসল কাজটুকু বুঝতে বাধ্য করা হলো
    prompt = f"""
    You are an AI assistant. Read the following developer's commit message and write a 2-line professional summary of what work was completed today. 
    DO NOT write anything else. Keep it strictly in English.
    
    Commit Message: {message}
    
    Summary of work completed:
    """
    
    print("⏳ Generating accurate report...")
    
    try:
        response = model.generate(prompt, max_tokens=100, temp=0.1)
        return response.strip()
    except Exception as e:
        return f"Error running local AI: {e}"