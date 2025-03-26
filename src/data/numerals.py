import re
import csv

file_path = "./ghu.hi"
output_path = "/Users/nawabkhan/Baby_Blue/GraphRAG_Project/data/processed/cleaned_1M(1).csv"

# Regex patterns

#lines part of some paragraph are numbered from  2 to whatever
start_number_pattern = re.compile(r'^\s*\d+\s+')  # Lines starting with "<space><number><space>"

# to handl patterns at start of lines like : 72: तीसरा क्लू: शाहिद का सरेंडर
colon_number_pattern = re.compile(r':\s*\d+\s*:')  # Matches ": <number> :"

#there are some roman numerals. removing them but not "I" (maybe the english word/letter)
roman_pattern = re.compile(r'\b(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b')  # Roman numerals

# Western → Hindi numeral mapping
western_to_hindi_map = str.maketrans("0123456789", "०१२३४५६७८९")

max_lines = 1_000_000  # Process first 1M lines
processed_lines = 0

with open(file_path, "r", encoding="utf-8") as file, open(output_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    for _ in range(max_lines):
        line = file.readline()
        if not line:
            break
        if start_number_pattern.match(line):  # Skip lines that start with a number
            continue
        cleaned_line = roman_pattern.sub(lambda x: "I" if x.group() == "I" else "", line).strip()  # Remove Roman numerals but keep "I"
        cleaned_line = colon_number_pattern.sub("", cleaned_line)  # Remove ": <number> :"
        cleaned_line = cleaned_line.translate(western_to_hindi_map)  # Convert Western numerals to Hindi
        writer.writerow([cleaned_line])
        processed_lines += 1

print(f"✅ Processed {processed_lines} lines. Saved cleaned text to 'cleaned_1M.csv'.")