import os

import pandas as pd
import convert_to_csv  # Step 1: Convert .hi to CSV
import remove_eng_lines  # Step 2: Remove English words
import numerals  # Step 3: Replace numerics with Hindi equivalents
import preprocess_data # Step 4 & 5

def preprocess_hindi_corpus(RAW_DATA_PATH, final_output, max_lines=10000):
    """Runs the complete preprocessing pipeline step by step."""
    
    # Step 1: Convert .hi file to CSV
    convert_to_csv.load_raw_data(RAW_DATA_PATH)

    # Step 2: Remove English words
    input1 = "hindi_corpus.csv" 
    output1 = "cleaned_output.csv"
    remove_eng_lines.remove_lines(input1, output1)

    # Step 3: Convert numerics (123 → १२३)
    output2 = "numeric_replaced.csv"
    numerals.replace_numerics(output1, output2)

    # Step 4: Remove stopwords
    raw_data = pd.read_csv(output2, nrows=max_lines)
    sentences = raw_data['text'].fillna("").astype(str).tolist()
    cleaned_sentences = [remove_stopwords_indic(line) for line in sentences]

    # Step 5: Tokenization
    tokenized_sentences = sentence_tokenize_text_grouped(cleaned_sentences)
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    pd.DataFrame({"text": tokenized_sentences}).to_csv(final_output, index=False, encoding='utf-8')        
    
    print(f"Preprocessing completed! Final output saved at: {final_output}")

# Example Usage
if __name__ == "__main__":
    RAW_DATA_PATH = "../monolingual/monolingual-n/" 
    final_output = "processed_corpus.csv"

    preprocess_hindi_corpus(final_output)
