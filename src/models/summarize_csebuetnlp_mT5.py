import pandas as pd
import re
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm



##to get data info
# nRowsRead = 10000 
# df = pd.read_csv('../../data/processed/tokenized_IITB.csv', delimiter=',', nrows = nRowsRead)
# df.dataframeName = 'test.csv'
# nRow, nCol = df.shape
# print(f'There are {nRow} rows and {nCol} columns')


# print(df.head())
# print(df.describe())
# print(df.info())


TOKENIZED_DATA_PATH = "../../data/processed/tokenized_IITB.csv"
SUMMARY_OUTPUT_PATH = "../../data/processed/summarized_IITB.csv"

df = pd.read_csv(TOKENIZED_DATA_PATH, nrows=100)
sentences = df['text'].fillna("").astype(str).tolist()

def group_sentences(sentences, chunk_size=5):
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def chunk_sentences(text, chunk_size=5):
    sentences = text.strip().split("|") 
    sentences = [s.strip() for s in sentences if s.strip()]
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

def summarize_chunks(chunks):
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






model_name = "csebuetnlp/mT5_multilingual_XLSum" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


final_summaries = []
grouped_chunks = group_sentences(sentences, chunk_size=5)
print(grouped_chunks)
final_summaries = summarize_chunks(grouped_chunks)

# === Save Summarized Output ===
os.makedirs(os.path.dirname(SUMMARY_OUTPUT_PATH), exist_ok=True)
pd.DataFrame({"summary": final_summaries}).to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding='utf-8')
print(f"✅ Summarized data saved to: {SUMMARY_OUTPUT_PATH}")


