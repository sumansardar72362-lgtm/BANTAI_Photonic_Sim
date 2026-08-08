from tracker import get_today_work
from word_report import add_daily_report


def main():
    print("Generating today's development report...")

    work = get_today_work()

    if not work:
        print("No work found for today.")
        return

    add_daily_report(work)

    print("Daily report added successfully.")


if __name__ == "__main__":
    main()