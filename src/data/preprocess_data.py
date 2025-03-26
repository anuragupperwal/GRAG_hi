import os
import pandas as pd
import re

from nltk.corpus import stopwords
from indicnlp.normalize import IndicNormalizerFactory
from transformers import AutoTokenizer
from indicnlp.tokenize import indic_tokenize


RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv" 
PROCESSED_DATA_PATH = "../../src/data/preocessed/processed_IITB.csv"
STOPWORDS_PATH = "/Users/anuragupperwal/Documents/Coding/ML/GRAG_hindi/src/data/src/data/external/stopwords-hi.txt"  # relative path to your stopwords file
TOKENIZED_OUTPUT_PATH = "../../data/processed/tokenized_IITB.csv"
CLEANED_OUTPUT_PATH = "../../data/processed/cleaned_IITB.csv"


raw_data = pd.read_csv(RAW_DATA_PATH, nrows=1000)
sentences = raw_data['text'].fillna("").astype(str).tolist()

# Optional: print first few lines. 
print("\nOriginal Data (first few lines):\n")
for idx, line in enumerate(raw_data['text'].head(5)):
    print(f"{idx}: {line}")


def load_indic_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f.readlines())
    return stopwords

indic_stopwords = load_indic_stopwords(STOPWORDS_PATH)

# -------------------------
# Remove Stopwords from Sentence
# -------------------------
def remove_stopwords_indic(text):
    words = text.split()
    return " ".join([word for word in words if word not in indic_stopwords])

# Apply Stopword Removal
cleaned_sentences = [remove_stopwords_indic(line) for line in sentences]


# # ---- Save Cleaned Data ----
# os.makedirs(os.path.dirname(CLEANED_OUTPUT_PATH), exist_ok=True)
# pd.DataFrame({"text": cleaned_sentences}).to_csv(CLEANED_OUTPUT_PATH, index=False, encoding="utf-8")
# print(f"✅ Cleaned data saved to: {CLEANED_OUTPUT_PATH}")

# ==== Word Tokenization ====
def word_tokenize_text(sentences):
    tokenized = []
    for sent in sentences:
        tokens = indic_tokenize.trivial_tokenize(sent, lang='hi')
        tokenized.append(" ".join(tokens))  # Re-join tokens for readability
    return tokenized

tokenized_sentences = word_tokenize_text(cleaned_sentences)

# ==== Save Tokenized Data ====
os.makedirs(os.path.dirname(TOKENIZED_OUTPUT_PATH), exist_ok=True)
pd.DataFrame({"text": tokenized_sentences}).to_csv(TOKENIZED_OUTPUT_PATH, index=False, encoding='utf-8')
print(f"✅ Tokenized data saved to: {TOKENIZED_OUTPUT_PATH}")