import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

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
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        input_length = len(tokenizer.encode(input_text, truncation=True, max_length=1024))
        max_output_length = max(30, int(input_length * 0.9))
        summary_ids = model.generate(
                        inputs, 
                        max_length=max_output_length,
                        min_length=int(max_output_length * 0.8),
                        length_penalty=0.9, 
                        num_beams=4, 
                        early_stopping=True
                    )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries



def summarize_corpus(input_path, output_path, chunk_size=4, max_lines=10000):
    """Main callable function to run summarization pipeline."""
    
    print(f"Loading tokenized input from: {input_path}")
    df = pd.read_csv(input_path, nrows=max_lines)
    sentences = df['text'].fillna("").astype(str).tolist()

    print(f"Loaded {len(df)} rows for summarization.")
    
    print("Grouping sentences...")
    grouped_chunks = group_sentences(sentences, chunk_size=chunk_size)

    print("Running summarization...")
    final_summaries = summarize_chunks(grouped_chunks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"summary": final_summaries}).to_csv(output_path, index=False, encoding='utf-8')

    print(f"Summarized data saved to: {output_path}")


# #Run directly for testing
# if __name__ == "__main__":
#     PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     TOKENIZED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_IITB.csv")
#     SUMMARY_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_IITB.csv")
#     summarize_corpus(
#         input_path=TOKENIZED_DATA_PATH,
#         output_path=SUMMARY_OUTPUT_PATH,
#         chunk_size=5,
#         max_lines=10000
#     )