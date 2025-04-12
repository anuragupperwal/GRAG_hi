import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def group_sentences(sentences, chunk_size=5):
    """Group N tokenized lines together for summarization."""
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def summarize_chunks(chunks):
    """Summarize each chunk using the loaded mT5 model."""
    summaries = []
    for chunk in tqdm(chunks, desc="Summarizing"):
        if not chunk.strip():
            summaries.append("")
            continue
        input_text = "summarize: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=100, min_length=15, length_penalty=1.5, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


def summarize_corpus(input_path, output_path, chunk_size=5, max_lines=100):
    """Main callable function to run summarization pipeline."""
    
    print(f"Loading tokenized input from: {input_path}")
    df = pd.read_csv(input_path, nrows=max_lines)
    sentences = df['text'].fillna("").astype(str).tolist()

    print("Grouping sentences...")
    grouped_chunks = group_sentences(sentences, chunk_size=chunk_size)

    print("Running summarization...")
    final_summaries = summarize_chunks(grouped_chunks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"summary": final_summaries}).to_csv(output_path, index=False, encoding='utf-8')

    print(f"Summarized data saved to: {output_path}")


#Run directly for testing
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    TOKENIZED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_IITB.csv")
    SUMMARY_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_IITB.csv")
    summarize_corpus(
        input_path=TOKENIZED_DATA_PATH,
        output_path=SUMMARY_OUTPUT_PATH,
        chunk_size=5,
        max_lines=100
    )