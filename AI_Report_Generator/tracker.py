from datetime import datetime
from pathlib import Path
from git import Repo


def find_git_repo():
    """
    AI_Report_Generator-এর parent folders থেকে
    আসল Git repository খুঁজে বের করবে।
    """

    current_folder = Path(__file__).resolve().parent

    for folder in [current_folder] + list(current_folder.parents):

        if (folder / ".git").exists():
            return Repo(folder)

    return None


def get_file_changes(repo):
    """
    Git working tree-তে বর্তমানে যেসব file পরিবর্তিত,
    সেগুলো বের করবে।
    """

    changes = []

    # Modified / deleted / renamed files
    for item in repo.index.diff(None):

        changes.append({
            "file": item.a_path,
            "status": "Modified"
        })

    # Untracked/new files
    for file_path in repo.untracked_files:

        changes.append({
            "file": file_path,
            "status": "New"
        })

    return changes


def get_today_commits(repo):
    """
    আজকের Git commits বের করবে।
    """

    today = datetime.now().date()

    commits = []

    for commit in repo.iter_commits():

        commit_date = datetime.fromtimestamp(
            commit.committed_date
        ).date()

        if commit_date == today:

            files = list(commit.stats.files.keys())

            commits.append({
                "message": commit.message.strip(),
                "files": files
            })

    return list(reversed(commits))


def get_diff_summary(repo):
    """
    বর্তমান uncommitted changes-এর statistics বের করবে।
    """

    try:

        diff = repo.git.diff(
            "--stat"
        )

        return diff.strip()

    except Exception:

        return ""


def get_today_work():

    try:

        repo = find_git_repo()

        if repo is None:

            print("Git repository not found.")

            return []

        work = []

        # --------------------------------
        # 1. Today's commits
        # --------------------------------

        commits = get_today_commits(repo)

        for commit in commits:

            work.append({
                "type": "commit",
                "commit": commit["message"],
                "files": commit["files"]
            })

        # --------------------------------
        # 2. Current uncommitted changes
        # --------------------------------

        current_changes = get_file_changes(repo)

        if current_changes:

            work.append({
                "type": "current_changes",
                "commit": "Current Project Changes",
                "files": [
                    change["file"]
                    for change in current_changes
                ]
            })

        # --------------------------------
        # 3. Diff statistics
        # --------------------------------

        diff_summary = get_diff_summary(repo)

        if diff_summary:

            work.append({
                "type": "diff",
                "commit": "Code Change Statistics",
                "files": [diff_summary]
            })

        return work

    except Exception as e:

        print("Git error:", e)

        return []