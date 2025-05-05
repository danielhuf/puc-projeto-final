import praw
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=f"script:aita_scraper:v1.0 (by /u/{os.getenv('REDDIT_USERNAME')})",
)


def get_submission_data(submission):
    """Extract relevant data from a submission."""
    return {
        "submission_id": submission.id,
        "url": submission.url,
        "created_utc": datetime.fromtimestamp(submission.created_utc),
        "text": submission.selftext,
        "score": submission.score,
    }


def get_comment_data(comment, submission_id):
    """Extract relevant data from a comment."""
    return {
        "submission_id": submission_id,
        "comment_id": comment.id,
        "author": str(comment.author),
        "text": comment.body,
        "score": comment.score,
        "created_utc": datetime.fromtimestamp(comment.created_utc),
    }


def fetch_submissions_and_comments(subreddit_name="AmItheAsshole", last_date=None):
    """Fetch submissions and their top-level comments after the last_date."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        subreddit.id
        print(f"Successfully connected to r/{subreddit_name}")
    except Exception as e:
        print(f"Error accessing subreddit: {str(e)}")
        return [], []

    submissions = []
    comments = []
    pbar = tqdm(
        total=1000, desc="Collecting submissions and comments", unit="submissions"
    )

    try:
        for submission in subreddit.new(limit=1000):
            if (
                last_date
                and datetime.fromtimestamp(submission.created_utc) <= last_date
            ):
                break

            submissions.append(get_submission_data(submission))
            submission.comments.replace_more(limit=None)

            for comment in submission.comments:
                if not comment.parent_id.startswith("t1_"):
                    comments.append(get_comment_data(comment, submission.id))

            pbar.update(1)

    except Exception as e:
        print(f"\nError while fetching data: {str(e)}")
        pbar.close()
        return submissions, comments

    pbar.close()
    return submissions, comments


def main():
    submissions_file = "aita_submissions.csv"
    comments_file = "aita_comments.csv"

    last_date = None
    if Path(submissions_file).exists():
        df_existing_submissions = pd.read_csv(submissions_file)
        if not df_existing_submissions.empty:
            last_date = pd.to_datetime(df_existing_submissions.iloc[0]["created_utc"])
            print(f"Last submission date: {last_date}")

    new_submissions, new_comments = fetch_submissions_and_comments(last_date=last_date)

    if new_submissions:
        df_new_submissions = pd.DataFrame(new_submissions)

        if Path(submissions_file).exists():
            df_combined_submissions = pd.concat(
                [df_new_submissions, df_existing_submissions], ignore_index=True
            )
        else:
            df_combined_submissions = df_new_submissions

        df_combined_submissions.to_csv(submissions_file, index=False)
        print(f"\nSaved {len(new_submissions)} new submissions to {submissions_file}")

        if new_comments:
            df_new_comments = pd.DataFrame(new_comments)

            if Path(comments_file).exists():
                df_existing_comments = pd.read_csv(comments_file)
                df_combined_comments = pd.concat(
                    [df_new_comments, df_existing_comments], ignore_index=True
                )
            else:
                df_combined_comments = df_new_comments

            df_combined_comments.to_csv(comments_file, index=False)
            print(f"Saved {len(new_comments)} new comments to {comments_file}")
        else:
            print("No new comments found.")
    else:
        print("No new submissions found.")


if __name__ == "__main__":
    main()

# https://github.com/Andrew-Sai/reddit-aita/tree/master
# https://github.com/iterative/aita_dataset/tree/master

# https://www.reddit.com/r/pushshift/comments/1c2ndiu/confused_on_how_to_use_pushshift/
# https://www.reddit.com/r/pushshift/comments/14ei799/pushshift_live_again_and_how_moderators_can/
# https://www.reddit.com/r/redditdev/comments/17fksud/get_posts_from_certain_dates_praw/
# https://www.reddit.com/r/pushshift/comments/148fv2n/not_able_to_retrieve_reddit_submissions_and/
