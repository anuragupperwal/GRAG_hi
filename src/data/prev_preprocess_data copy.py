import os
import gzip
import pandas as pd
import re

from indicnlp.normalize import IndicNormalizerFactory
from transformers import AutoTokenizer


RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv" 
PROCESSED_DATA_PATH = "../../src/data/preocessed/processed_IITB.csv"


raw_data = pd.read_csv(RAW_DATA_PATH, nrows=1000)
# print(raw_data['text'])

#to print raw data line by line
for idx, line in enumerate(raw_data['text']):
    print(f"{idx}: {line}")

cleaned_data=""
for i in raw_data['text']:
    i = i.strip()
    if re.fullmatch(r"[0-9]+", i):
        continue 

    cleaned_line = re.sub(r"[^A-Za-z0-9\u0900-\u097F\s!]", "", i) #this step remove everything except numbers, space, !, hindi words. But we are missing the english words that we have inbetween
    cleaned_data += cleaned_line 

print(cleaned_data)


# #to remove non-hindi char, remove extra space and puntuations, eng letters
# def clean_data(data):
#     cleaned_data=""
#     for i in data:
#         i = i.strip()
#         if re.fullmatch(r"[0-9]+", i):
#             continue 

#         cleaned_line = re.sub(r"[^0-9\u0900-\u097F\s!]", "", i)
#         cleaned_data += cleaned_line


#     return cleaned_data

# # sample_text = ["महात्मा गांधी का जन्म 1869 में हुआ था! @Gandhi was born in 1869."]
# # cleaned_sample = clean_text(raw_data[:15])
# # print("Before cleaning: ", raw_data[:15])
# # print("After cleaning: ", cleaned_sample)


# #idk kya hi normalise ho rha h
# #normalise data
# def normalise_data(data):
#     normalizer = IndicNormalizerFactory().get_normalizer("hi")
    
#     normalized = normalizer.normalize(data)  # Normalize text

#     return normalized

# # normalise_data = normalise_data(cleaned_sample)
# # print("Before normalized: ", cleaned_sample)
# # print("After normalized: ", normalise_data)


# def preprocess_data(data):
#     data = clean_data(data)  # Improved cleaning
#     data = normalise_data(data)  # Standardized normalization
#     return data

# # Apply preprocessing to all loaded data
# print("Working: ")
# # cleaned_data = [preprocess_data(line) for line in raw_data[:200]]
# # print("After everything: ", cleaned_data[:5])  # Print first 5 cleaned sentences
# print("Done!")



# # print(cleaned_data[0])

# #2191100


# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)
# def tokenize_subwords(data):
#     tokenized_data = []
#     for sentence in data:
#         tokens = tokenizer.tokenize(sentence)
#         tokenized_data.append(tokens)
#     return tokenized_data
    
# tokenized_data = tokenize_subwords(cleaned_data)
# print(tokenized_data[:5000])



