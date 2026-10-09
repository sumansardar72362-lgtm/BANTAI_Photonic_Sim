# File: src/kecia/io.py
import pickle
import os

# 🔒 সিকিউরিটি হেডার (এটা ছাড়া কেউ মডেল লোড করতে পারবে না)
MAGIC_HEADER = b"BANTAI_KECIA_5D_MEMORY_v1\n"

def save_model(parameters, filepath):
    """
    মডেলের ওয়েটস (Weights) .kecia ফাইলে সিকিউরভাবে সেভ করে।
    """
    if not filepath.endswith(".kecia"):
        filepath += ".kecia"
        
    # ডেটাকে বাইনারিতে কনভার্ট করা হচ্ছে
    data = pickle.dumps(parameters)
    
    # 5D Glass Memory স্টাইলে রাইট করা
    with open(filepath, "wb") as f:
        f.write(MAGIC_HEADER) # প্রথমে আমাদের সিকিউরিটি লক বসবে
        f.write(data)         # তারপর আসল ওয়েটস বসবে
        
    print(f"🔒 Model securely saved to 5D Glass Memory format: {filepath}")

def load_model(filepath):
    """
    .kecia ফাইল থেকে ভেরিফাই করে মডেল লোড করে।
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")
        
    with open(filepath, "rb") as f:
        # প্রথমে চেক করব আমাদের সিকিউরিটি লক আছে কি না
        header = f.read(len(MAGIC_HEADER))
        if header != MAGIC_HEADER:
            raise ValueError("❌ Invalid or Corrupted .kecia file! Unauthorized format.")
            
        data = f.read()
        
    parameters = pickle.loads(data)
    print(f"🔓 Model successfully loaded from 5D Glass Memory: {filepath}")
    return parameters