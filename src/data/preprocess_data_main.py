import os
import pandas as pd
from data import remove_eng_lines, numerals
from indicnlp.tokenize.sentence_tokenize import sentence_split

# === CONFIG ===
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "stopwords-hi.txt")

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


def preprocess_hindi_corpus(RAW_DATA_PATH, max_lines=10000, stopwords_path=STOPWORDS_PATH, project_root=None):
    """Runs the complete preprocessing pipeline step by step."""

    #absolute paths
    input1 = RAW_DATA_PATH
    print("this is the test print to check project_root", project_root)
    output1 = os.path.join(project_root, "data/raw/cleaned_output.csv")
    output2 = os.path.join(project_root, "data/raw/numeric_replaced.csv")
    final_output = os.path.join(project_root, "data/processed/tokenized_IITB.csv")    


    #Step 1
    # raw_data = pd.read_csv(input1, nrows=max_lines)
    # read 100 lines after evrey 5000 lines
    chunk_size = 5000
    sample_size = 100
    dfs = []
    total_rows_read = 0
    while total_rows_read < max_lines:
        skip_rows = list(range(1, total_rows_read + chunk_size - sample_size + 1))
        try:
            chunk = pd.read_csv(input1, skiprows=skip_rows, nrows=sample_size)
            dfs.append(chunk)
            total_rows_read += chunk_size
        except pd.errors.EmptyDataError:
            break
    raw_data = pd.concat(dfs, ignore_index=True)


    # Step 2: Remove English lines
    os.makedirs(os.path.dirname(output1), exist_ok=True)  # Ensure directory exists
    remove_eng_lines.remove_lines(input1, output1, max_lines=max_lines)

    # Step 3: Convert numerics (123 → १२३)
    os.makedirs(os.path.dirname(output2), exist_ok=True) 
    numerals.replace_numerics(output1, output2, max_lines=max_lines)

    # Step 4: Stopword removal
    raw_data = pd.read_csv(output2, nrows=max_lines)
    sentences = raw_data['text'].fillna("").astype(str).tolist()
    # stopwords = load_indic_stopwords(stopwords_path)
    # cleaned_sentences = [remove_stopwords_indic(line, stopwords) for line in sentences]
    cleaned_sentences = sentences
    
    # Step 5: Sentence tokenization
    tokenized_sentences = sentence_tokenize_text_grouped(cleaned_sentences)
    joined_sentences = [" ".join(sent_list) for sent_list in tokenized_sentences]

    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    pd.DataFrame({"text": joined_sentences}).to_csv(final_output, index=False, encoding='utf-8')
    print(f"Preprocessing completed! Final output saved at: {final_output}")

    # Optional Step 6: Pie Chart
    # cleaned_lines = len([line for line in sentences if line.strip() and len(line.strip()) > 5])
    # plot_cleaning_results(total_lines, cleaned_lines, CHART_OUTPUT_PATH)


# === Entry Point ===
# if __name__ == "__main__":
#     RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "data/raw/monolingual-n/raw_IITB.csv")
#     RAW_DATA_PATH = os.path.abspath(RAW_DATA_PATH)  # Convert to absolute path

#     print(RAW_DATA_PATH*5)
#     preprocess_hindi_corpus(RAW_DATA_PATH, 100)
