import os
import pandas as pd
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
# from indicnlp.normalize import IndicNormalizerFactory
from transformers import AutoTokenizer
from indicnlp.tokenize.sentence_tokenize import sentence_split

RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv" 
PROCESSED_DATA_PATH = "../../src/data/preocessed/processed_IITB.csv"
STOPWORDS_PATH = "./stopwords-hi.txt" 
TOKENIZED_OUTPUT_PATH = "../../data/processed/tokenized_IITB.csv"
CLEANED_OUTPUT_PATH = "../../data/processed/cleaned_IITB.csv"
CHART_OUTPUT_PATH = "../../data/Screenshotcleaning_pie_chart.png"


raw_data = pd.read_csv(RAW_DATA_PATH, nrows=100)
sentences = raw_data['text'].fillna("").astype(str).tolist()


print("\nOriginal Data (first few lines):\n")
for idx, line in enumerate(raw_data['text'].head(5)):
    print(f"{idx}: {line}")


def load_indic_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f.readlines())
    return stopwords

def remove_stopwords_indic(text):
    words = text.split()
    return " ".join([word for word in words if word not in indic_stopwords])

def sentence_tokenize_text_grouped(sentences):
    tokenized = []
    for sent in sentences:
        sents = sentence_split(sent, lang='hi')
        tokenized.append(sents)  # list of lists
    return tokenized





# def plot_cleaning_results(total_lines: int, cleaned_lines: int, save_path: str = "../../reports/figures/cleaning_pie_chart.png"):
#     removed_lines = total_lines - cleaned_lines
#     labels = ['Cleaned & Retained', 'Removed as Noise/Junk']
#     sizes = [cleaned_lines, removed_lines]
#     explode = (0.05, 0.1)  # slight explosion for visual impact
#     colors = ['#4CAF50', '#F44336']

#     plt.figure(figsize=(6, 6))
#     plt.pie(sizes, explode=explode, labels=labels, colors=colors,
#             autopct='%1.1f%%', shadow=True, startangle=140)
#     plt.title('🧹 Cleaning Results: Hindi Monolingual Dataset')
#     plt.axis('equal')
    
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()
#     print(f"✅ Pie chart saved to {save_path}")




indic_stopwords = load_indic_stopwords(STOPWORDS_PATH)
cleaned_sentences = [remove_stopwords_indic(line) for line in sentences]
tokenized_sentences = sentence_tokenize_text_grouped(cleaned_sentences)


print("\nbefore saving (first few lines):\n")
for idx, line in enumerate(tokenized_sentences[:5]):
    print(f"{idx}: {line}")

#for plotting pie chart 
# raw_df = pd.read_csv(RAW_DATA_PATH)
# clean_tokenised_df = pd.read_csv(TOKENIZED_OUTPUT_PATH)

# #Count Valid Sentences
# total_lines = len(raw_df)
# cleaned_lines = len(cleaned_df)

#Plot
# plot_cleaning_results(total_lines, cleaned_lines, CHART_OUTPUT_PATH)

#Save Tokenized Data
os.makedirs(os.path.dirname(TOKENIZED_OUTPUT_PATH), exist_ok=True)
joined_sentences = [" <s> ".join(sent_list) for sent_list in tokenized_sentences]
pd.DataFrame({"text": joined_sentences}).to_csv(TOKENIZED_OUTPUT_PATH, index=False, encoding='utf-8')
print(f"✅ Tokenized data saved to: {TOKENIZED_OUTPUT_PATH}")