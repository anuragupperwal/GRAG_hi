import re
import pandas as pd

def replace_numerics(input_csv, output_csv, text_column="text", max_lines=10000):
    """
    Cleans a Hindi text corpus stored in a CSV file by:
    - Removing numbered paragraph indicators
    - Removing ": <number> :" patterns
    - Removing Roman numerals (except 'I')
    - Converting Western numerals (0-9) to Hindi numerals (०-९)
    - Keeping only the first `max_rows` lines
    """

    # Regex patterns
    start_number_pattern = re.compile(r'^\s*\d+\s+')  # Lines starting with "<space><number><space>"
    colon_number_pattern = re.compile(r':\s*\d+\s*:')  # Matches ": <number> :"
    roman_pattern = re.compile(r'\b(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b')  # Roman numerals
    western_to_hindi_map = str.maketrans("0123456789", "०१२३४५६७८९")  # Western → Hindi numeral mapping


    df = pd.read_csv(input_csv, encoding="utf-8", nrows=max_lines)
    def clean_text(text):
        if pd.isna(text):  # Handle NaN values
            return ""
        text = str(text)  # Ensure string format
        if start_number_pattern.match(text):  # Skip lines that start with a number
            return ""
        text = roman_pattern.sub(lambda x: "I" if x.group() == "I" else "", text).strip()  # Remove Roman numerals but keep "I"
        text = colon_number_pattern.sub("", text)  # Remove ": <number> :"
        text = text.translate(western_to_hindi_map)  # Convert Western numerals to Hindi
        return text

    df[text_column] = df[text_column].apply(clean_text)
    df = df[df[text_column] != ""]
    df.to_csv(output_csv, index=False, encoding="utf-8")