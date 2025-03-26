import re
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def detect_english_words(text):
    return re.findall(r"\b[a-zA-Z]+\b", text)


translation_cache = {}

def translate_word_en_to_hi(word):
    """Translates a single English word to Hindi using the model."""
    if word in translation_cache:
        return translation_cache[word]
    
    input_text = f"translate English to Hindi: {word}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_length=40)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    translation_cache[word] = translated
    return translated

def translate_english_words_in_sentence(sentence):
    """Detects English words in a sentence and replaces them with their Hindi translations."""
    english_words = detect_english_words(sentence)
    for word in english_words:
        hindi_word = translate_word_en_to_hi(word)
        sentence = sentence.replace(word, hindi_word)
    return sentence


model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

#for tokenization automatically based on the model being used. ai4bharat have a custom tokenizer. But a few eg generally used are MBartTokenizer, T5Tokenizer, etc
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
#transformer model for tranalation/summarization/QA
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to("cpu") 


sample = "मुझे Apple और Google बहुत पसंद हैं।"
translated = translate_english_words_in_sentence(sample)
print(translated)


