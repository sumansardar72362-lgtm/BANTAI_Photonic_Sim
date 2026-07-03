import os

def list_files(startpath):
    print("\n--- 🌳 TOMAR CURRENT PROJECT STRUCTURE 🌳 ---\n")
    for root, dirs, files in os.walk(startpath):
        # .venv বা অন্যান্য অপ্রয়োজনীয় ফোল্ডারগুলো বাদ দেওয়া হচ্ছে যাতে লিস্ট পরিষ্কার থাকে
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '    ' * level
        folder_name = os.path.basename(os.path.abspath(root))
        
        if level == 0:
            print(f"📁 {folder_name}/ (Main Folder)")
        else:
            print(f"{indent}📂 {folder_name}/")
        
        subindent = '    ' * (level + 1)
        for f in files:
            print(f"{subindent}📄 {f}")
    print("\n---------------------------------------------\n")

# কোড রান করা হচ্ছে
list_files('.')