import os
import pandas as pd
import convert_to_csv
import remove_eng_lines
import numerals
import matplotlib.pyplot as plt
from indicnlp.tokenize.sentence_tokenize import sentence_split

# === CONFIG ===
STOPWORDS_PATH = "./stopwords-hi.txt"
CHART_OUTPUT_PATH = "../../reports/figures/cleaning_pie_chart.png"


def load_indic_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f.readlines())
    return stopwords


def remove_stopwords_indic(text, stopwords):
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])


def sentence_tokenize_text_grouped(sentences):
    tokenized = []
    for sent in sentences:
        sents = sentence_split(sent, lang='hi')
        sents = [s.replace('"', '').strip() for s in sents if s.strip()]
        tokenized.append(sents)
    return tokenized


def plot_cleaning_results(total_lines: int, cleaned_lines: int, save_path: str):
    removed_lines = total_lines - cleaned_lines
    labels = ['Cleaned & Retained', 'Removed as Noise/Junk']
    sizes = [cleaned_lines, removed_lines]
    explode = (0.05, 0.1)
    colors = ['#4CAF50', '#F44336']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('🧹 Cleaning Results: Hindi Monolingual Dataset')
    plt.axis('equal')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Pie chart saved to {save_path}")


def preprocess_hindi_corpus(RAW_DATA_PATH, final_output, max_lines=10000):
    """Runs the complete preprocessing pipeline step by step."""

    # Step 1: Convert .hi file to CSV
    # convert_to_csv.load_raw_data(RAW_DATA_PATH)
    raw_data = pd.read_csv(RAW_DATA_PATH, nrows=100)

    # Step 2: Remove English lines
    input1 = "../../data/raw/monolingual-n/raw_IITB.csv"
    output1 = "../../data/raw/monolingual-n/cleaned_output.csv"
    remove_eng_lines.remove_lines(input1, output1)

    # Step 3: Convert numerics (123 → १२३)
    output2 = "../../data/raw/monolingual-n/numeric_replaced.csv"
    numerals.replace_numerics(output1, output2)

    # Step 4: Stopword removal
    raw_data = pd.read_csv(output2, nrows=max_lines)
    total_lines = len(raw_data)
    sentences = raw_data['text'].fillna("").astype(str).tolist()
    stopwords = load_indic_stopwords(STOPWORDS_PATH)
    cleaned_sentences = [remove_stopwords_indic(line, stopwords) for line in sentences]

    # Step 5: Sentence tokenization
    tokenized_sentences = sentence_tokenize_text_grouped(cleaned_sentences)
    joined_sentences = [" <s> ".join(sent_list) for sent_list in tokenized_sentences]

    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    pd.DataFrame({"text": joined_sentences}).to_csv(final_output, index=False, encoding='utf-8')
    print(f"✅ Preprocessing completed! Final output saved at: {final_output}")

    # Optional Step 6: Pie Chart
    # cleaned_lines = len([line for line in sentences if line.strip() and len(line.strip()) > 5])
    # plot_cleaning_results(total_lines, cleaned_lines, CHART_OUTPUT_PATH)


# === Entry Point ===
if __name__ == "__main__":
    RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv"
    final_output = "../../data/processed/processed_corpus.csv"
    preprocess_hindi_corpus(RAW_DATA_PATH, final_output, 100)
