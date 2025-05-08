import pandas as pd
import re


def get_verdict(text):
    text = text.lower()

    patterns = {
        "YTA": r"\b(yta|you\'?re\s+the\s+asshole|you\s+are\s+the\s+asshole)\b",
        "NTA": r"\b(nta|not\s+the\s+asshole|not\s+an\s+asshole)\b",
        "NAH": r"\b(nah|no\s+assholes\s+here|no\s+asshole\s+here)\b",
        "ESH": r"\b(esh|everyone\s+sucks\s+here|everybody\s+sucks\s+here)\b",
        "INFO": r"\b(info|more\s+information\s+needed|need\s+more\s+info)\b",
    }

    for verdict, pattern in patterns.items():
        if re.search(pattern, text):
            return verdict
    return None


def get_top_comments(df_comments):
    top_comments = (
        df_comments.sort_values("score", ascending=False)
        .groupby("submission_id")
        .first()
    )
    return top_comments


def process_data():
    df_submissions = pd.read_csv("aita_submissions.csv")
    print(f"Total number of submissions: {len(df_submissions)}")

    df_comments = pd.read_csv("aita_comments.csv")
    print(f"Total number of comments: {len(df_comments)}\n")

    df_submissions = df_submissions[df_submissions["score"] >= 25]
    print(f"Number of submissions with score >= 25: {len(df_submissions)}")

    df_submissions = df_submissions[df_submissions["text"].str.len() >= 300]
    print(f"Number of submissions with text length >= 300: {len(df_submissions)}")

    df_comments = df_comments[
        ~df_comments["author"].isin(["Judgement_Bot_AITA", "AutoModerator"])
    ]
    print(f"Number of comments after removing bot comments: {len(df_comments)}")

    # Add verdict column and filter out comments without verdicts
    df_comments["verdict"] = df_comments["text"].apply(get_verdict)
    df_comments = df_comments[df_comments["verdict"].notna()]
    print(f"Number of comments containing a verdict: {len(df_comments)}")

    top_comments = get_top_comments(df_comments)
    valid_submissions = top_comments[top_comments["text"].str.len() >= 25].index

    df_submissions = df_submissions[
        df_submissions["submission_id"].isin(valid_submissions)
    ]
    print(
        f"Number of submissions with top comments length >= 25: {len(df_submissions)}"
    )

    df_submissions = df_submissions.merge(
        top_comments[["comment_id"]],
        left_on="submission_id",
        right_index=True,
        how="left",
    )

    df_comments = df_comments[
        df_comments["submission_id"].isin(df_submissions["submission_id"])
    ]
    print(f"Number of comments after removing orphan comments: {len(df_comments)}")

    print(f"\nTotal number of submissions after filtering: {len(df_submissions)}")
    print(f"Total number of comments after filtering: {len(df_comments)}\n")

    return df_submissions, df_comments


if __name__ == "__main__":
    process_data()
