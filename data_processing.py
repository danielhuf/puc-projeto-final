import pandas as pd


def process_submissions():
    df = pd.read_csv("aita_submissions.csv")
    print(f"Total number of submissions: {len(df)}")

    df_filtered = df[df["score"] >= 25]
    print(f"Number of submissions with score >= 25: {len(df_filtered)}")

    df_filtered = df_filtered[df_filtered["text"].str.len() >= 300]
    print(
        f"Number of submissions with score >= 25 and text length >= 300: {len(df_filtered)}"
    )

    return df_filtered


def process_comments():
    df_comments = pd.read_csv("aita_comments.csv")
    print(f"\nTotal number of comments: {len(df_comments)}")

    df_filtered = df_comments[
        ~df_comments["author"].isin(["Judgement_Bot_AITA", "AutoModerator"])
    ]
    print(f"Number of comments after removing bot comments: {len(df_filtered)}")

    return df_filtered


if __name__ == "__main__":
    process_submissions()
    process_comments()
