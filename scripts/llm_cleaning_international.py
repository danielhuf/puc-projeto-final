import pandas as pd
import os
import re

COLUMNS_TO_CHECK = [
    "gpt3.5_reason_1",
    "gpt3.5_reason_2",
    "gpt4_reason_1",
    "gpt4_reason_2",
    "claude_reason_1",
    "claude_reason_2",
    "gemini_reason_1",
    "gemini_reason_2",
    "llama_reason_1",
    "llama_reason_2",
    "mistral_reason_1",
    "mistral_reason_2",
    "gemma_reason_1",
    "gemma_reason_2",
]

CSV_FILES = [
    "ethical_dilemmas_br.csv",
    "ethical_dilemmas_de.csv",
    "ethical_dilemmas_es.csv",
    "ethical_dilemmas_fr.csv",
]

COLUMNS_TO_CLEAN = [
    "llama_reason_1",
    "llama_reason_2",
    "mistral_reason_1",
    "mistral_reason_2",
    "gemma_reason_1",
    "gemma_reason_2",
]


def clean_text(text):
    """
    Clean text by removing unwanted words, symbols, and formatting.

    Args:
        text: string to clean

    Returns:
        Cleaned plain text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return text

    # Remove brackets: {}, [], ()
    text = re.sub(r"[{}\[\]()]", "", text)

    # Remove markdown indicators: **, ##, ```, #
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"#+", "", text)

    # Remove "Objeto", "JSON" and similar prefixes (case-insensitive)
    text = re.sub(r"\bobje[ct]o\s*json\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bjson\b", "", text, flags=re.IGNORECASE)

    # Remove "verdict", "verdicte", "veredito" (case-insensitive - multiple languages)
    text = re.sub(r"\bvere?d[iu][ct]te?o?\b", "", text, flags=re.IGNORECASE)

    # Remove "reasoning" and variations like "reasonning" (case-insensitive)
    text = re.sub(r"\breason[ni]+ng?\b", "", text, flags=re.IGNORECASE)

    # Remove specific verdict labels: YTA, NTA, ESH, NAH, INFO (case-insensitive)
    text = re.sub(r"\b(YTA|NTA|ESH|NAH|INFO)\b", "", text, flags=re.IGNORECASE)

    # Remove all types of quotation marks (straight and curly) and colons
    text = re.sub(r'["\'\u201c\u201d\u2018\u2019:]', "", text)

    # Remove common punctuation artifacts at the beginning/end
    text = re.sub(r"^[\s,;]+", "", text)
    text = re.sub(r"[\s,;]+$", "", text)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_dataframe(df, filename):
    """
    Clean a dataframe by removing rows where any reason column has a null value
    and cleaning malformed text in specific columns.

    Args:
        df: pandas DataFrame to clean
        filename: name of the file (for display purposes)

    Returns:
        Cleaned pandas DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")

    existing_columns = [col for col in COLUMNS_TO_CHECK if col in df.columns]
    df_cleaned = df.dropna(subset=existing_columns).copy()

    columns_to_clean = [col for col in COLUMNS_TO_CLEAN if col in df_cleaned.columns]
    for col in columns_to_clean:
        df_cleaned[col] = df_cleaned[col].apply(clean_text)

    final_rows = len(df_cleaned)
    print(f"Final number of rows: {final_rows}")

    return df_cleaned


def main():
    """Main function to process all CSV files."""
    data_folder = "data"

    for csv_file in CSV_FILES:
        file_path = os.path.join(data_folder, csv_file)

        if not os.path.exists(file_path):
            print(f"\nWarning: File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        df_cleaned = clean_dataframe(df, csv_file)

        parts = csv_file.replace(".csv", "").rsplit("_", 1)
        output_filename = f"{parts[0]}_cleaned_{parts[1]}.csv"
        output_path = os.path.join(data_folder, output_filename)
        df_cleaned.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
