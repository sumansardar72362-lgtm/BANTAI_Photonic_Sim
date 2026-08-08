from datetime import datetime
from git import Repo


def get_today_work():
    try:
        repo = Repo(".")

        today = datetime.now().date()

        commits = list(repo.iter_commits())

        today_commits = []

        for commit in commits:
            commit_date = datetime.fromtimestamp(
                commit.committed_date
            ).date()

            if commit_date == today:
                today_commits.append(commit)

        if not today_commits:
            return []

        work = []

        for commit in reversed(today_commits):

            files = list(commit.stats.files.keys())

            work.append({
                "commit": commit.message.strip(),
                "files": files
            })

        return work

    except Exception as e:
        print("Git error:", e)
        return []