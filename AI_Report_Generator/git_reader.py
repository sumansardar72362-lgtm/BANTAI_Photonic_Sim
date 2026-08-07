import git
import os

def get_latest_commit_data():
    """
    মেইন প্রজেক্টের গিট রিপোজিটরি থেকে লেটেস্ট কমিট মেসেজ এবং কোড চেঞ্জ (diff) বের করে আনে।
    """
    # আমাদের মেইন প্রজেক্টের পাথ (AI_Report_Generator ফোল্ডারের এক ধাপ বাইরে)
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    try:
        repo = git.Repo(repo_path)
        
        # যদি এখনো কোনো কমিট না হয়ে থাকে
        if not repo.heads:
            return None, None, "No commits found yet. Please make your first commit."
            
        latest_commit = repo.head.commit
        
        # কমিটের বেসিক তথ্য
        date = latest_commit.committed_datetime.strftime("%d %B %Y")
        message = latest_commit.message.strip()
        
        # কোডের পরিবর্তন (Diff) বের করা
        if not latest_commit.parents:
            # এটিই যদি প্রথম কমিট হয়
            diff = repo.git.show(latest_commit.hexsha)
        else:
            # আগের কমিটের সাথে বর্তমান কমিটের পার্থক্য
            diff = repo.git.diff(latest_commit.parents[0].hexsha, latest_commit.hexsha)
            
        return date, message, diff
        
    except git.exc.InvalidGitRepositoryError:
        return None, None, "Error: Git is not initialized in the main project directory."
    except Exception as e:
        return None, None, f"An error occurred: {str(e)}"