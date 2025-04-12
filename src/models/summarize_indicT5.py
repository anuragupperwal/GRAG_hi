import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --------- Paths ---------
TOKENIZED_INPUT_PATH = "../../data/processed/tokenized_IITB.csv"
SUMMARY_OUTPUT_PATH = "../../data/processed/summarized_IITB.csv"
MODEL_NAME = "ai4bharat/IndicT5-Generation"

# --------- Load Model & Tokenizer ---------
print("🔄 Loading IndicT5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --------- Load Tokenized Data ---------
print("📥 Loading tokenized data...")
df = pd.read_csv(TOKENIZED_INPUT_PATH)
sentences = df['text'].fillna("").astype(str).tolist()

# --------- Utility: Group N Sentences Together ---------
def group_sentences(sent_list, group_size=3):
    grouped = []
    for i in range(0, len(sent_list), group_size):
        chunk = " ".join(sent_list[i:i + group_size])
        grouped.append(chunk)
    return grouped

# --------- Summarization Function ---------
def summarize_text(text_list, max_input_length=512, max_summary_length=10000):
    summaries = []
    for chunk in tqdm(text_list, desc="🧠 Generating summaries"):
        if not chunk.strip():
            summaries.append("")
            continue
        try:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
            summary_ids = model.generate(inputs.input_ids, max_length=max_summary_length, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("[ERROR]")
    return summaries

# --------- Pipeline ---------
print("🔗 Grouping tokenized sentences...")
grouped_chunks = group_sentences(sentences, group_size=3)  # You can change group_size=5 if needed

print("📝 Running summarization...")
summaries = summarize_text(grouped_chunks)

# --------- Save Output ---------
print("💾 Saving summarized output...")
os.makedirs(os.path.dirname(SUMMARY_OUTPUT_PATH), exist_ok=True)
pd.DataFrame({"summary": summaries}).to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding='utf-8')
print(f"✅ Summarized output saved to: {SUMMARY_OUTPUT_PATH}")