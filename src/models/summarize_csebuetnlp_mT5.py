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


# === File Paths ===
TOKENIZED_DATA_PATH = "../../data/processed/tokenized_IITB.csv"
SUMMARY_OUTPUT_PATH = "../../data/processed/summarized_IITB.csv"

# # === Load Tokenized Data ===
# print("🔍 Loading tokenized data...")
df = pd.read_csv(TOKENIZED_DATA_PATH, nrows=100)

texts = df['text'].fillna("").astype(str).tolist()

# model_name = "csebuetnlp/mT5_multilingual_XLSum" #to load model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# def chunk_sentences(text, chunk_size=3):
#     # If text is a list in string form (e.g., "['a', 'b']"), convert it to list
#     try:
#         sentences = eval(text)
#     except:
#         return []
    
#     return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

# def summarize_chunks(chunks):
#     summaries = []
#     for chunk in chunks:
#         if not chunk.strip():
#             summaries.append("")
#             continue

#         input_text = "summarize: " + chunk
#         inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
#         summary_ids = model.generate(inputs, max_length=100, min_length=15, length_penalty=1.5, num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)
#     return summaries

# # === Process All Rows ===
# final_summaries = []

# print("🧠 Generating summaries...")
# for line in tqdm(texts):
#     chunks = chunk_sentences(line, chunk_size=3)
#     summary_list = summarize_chunks(chunks)
#     final_summary = " ".join(summary_list)  # Join all summaries of chunks
#     final_summaries.append(final_summary)

# # === Save Summarized Output ===
# os.makedirs(os.path.dirname(SUMMARY_OUTPUT_PATH), exist_ok=True)
# pd.DataFrame({"summary": final_summaries}).to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding='utf-8')

# print(f"✅ Summarized data saved to: {SUMMARY_OUTPUT_PATH}")



# ask gpt
#   File "/Users/anuragupperwal/Documents/Coding/ML/GRAG_hi/graphrag_env/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 87, in import_protobuf_decode_error
#     raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))
# ImportError: 
#  requires the protobuf library but it was not found in your environment. Checkout the instructions on the
# installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
# that match your environment. Please note that you may need to restart your runtime after installation.


# while running the code you gave me for csebuetnlp/mT5_multilingual_XLSum this model


article_text = ""
# WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
for line in tqdm(texts):
    article_text += line

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    article_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)