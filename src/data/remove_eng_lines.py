import re
import pandas as pd

def contains_english(text):
    return bool(re.search(r"\b[a-zA-Z]+\b", str(text)))

def remove_lines(input_csv, output_csv, text_column="text", max_lines=10000):
    """Removes rows containing English words from a specific column in the CSV."""

    df = pd.read_csv(input_csv, encoding="utf-8", nrows=max_lines)  
    df_cleaned = df[~df[text_column].apply(contains_english)]  
    df_cleaned.to_csv(output_csv, index=False, encoding="utf-8")  