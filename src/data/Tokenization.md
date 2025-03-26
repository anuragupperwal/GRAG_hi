
### 2 Methods :

# ==1. If Speed over Accuracy, Use indic-nlp-library==

from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from indicnlp.morph import unsupervised_morph

# Load Hindi Stopwords
def load_stopwords(filepath="stopwords_hi.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())
    return stopwords

# Stopword Removal
def remove_stopwords(words, stopwords):
    return [[word for word in sentence if word not in stopwords] for sentence in words]

# Simple Hindi Stemming (Rule-Based Suffix Stripping)
def stem_word(word):
    suffixes = ["ा", "ी", "े", "ों", "ो", "ना", "ने", "कर", "ता", "ती", "ते", "ीय"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# Apply Stemming
def apply_stemming(words):
    return [[stem_word(word) for word in sentence] for sentence in words]

# Hindi Lemmatization using Morphological Analysis
def lemmatize_word(word):
    analyses = unsupervised_morph.Analyzer().analyze(word)
    # If multiple root forms, choose the shortest one (better approximation)
    lemmas = [analysis[0] for analysis in analyses]
    if lemmas:
        return min(lemmas, key=len)  
    return word  # Return original if no analysis found

# Apply Lemmatization
def apply_lemmatization(words):
    return [[lemmatize_word(word) for word in sentence] for sentence in words]

# Transliteration (Devanagari → Roman)
def transliterate(text, src="hi", target="en"):
    return UnicodeIndicTransliterator.transliterate(text, src, target)

# Tokenization Function
def tokenize_indic(text, lang="hi"):
    sentences = sentence_tokenize.sentence_split(text, lang)
    words = [indic_tokenize.trivial_tokenize(sent) for sent in sentences]
    return sentences, words

# Example Text
text = "क्रिप्टोग्राफी एक महत्वपूर्ण विषय है। यह सूचना सुरक्षा में उपयोगी है।"

# Load Stopwords
stopwords = load_stopwords()

# Step 1: Tokenization
sentences, words = tokenize_indic(text)

# Step 2: Stopword Removal
filtered_words = remove_stopwords(words, stopwords)

# Step 3: Lemmatization
lemmatized_words = apply_lemmatization(filtered_words)

# Step 4: Transliteration (for better readability)
transliterated_text = transliterate(text)

# Print Results
print("\n🔹 Original Text:")
print(text)

print("\n🔹 Sentence Tokenization:")
for sent in sentences:
    print(f"- {sent}")

print("\n🔹 Word Tokenization:")
for i, word_list in enumerate(words):
    print(f"Sentence {i+1}: {word_list}")

print("\n🔹 After Stopword Removal:")
for i, word_list in enumerate(filtered_words):
    print(f"Sentence {i+1}: {word_list}")

print("\n🔹 After Lemmatization:")
for i, word_list in enumerate(lemmatized_words):
    print(f"Sentence {i+1}: {word_list}")

print("\n🔹 Transliteration (Devanagari → Roman):")
print(transliterated_text)

# ==2. Best of both Worlds (Accuracy : max, Speed : meh)==

**Install Dependencies**
pip install spacy spacy-udpipe indic-nlp-library

**Download the Hindi UD model for spaCy**
python -m spacy download xx_ent_wiki_sm
**Or Manually**
spacy_udpipe.download("hi")


import spacy
import spacy_udpipe
from indicnlp.tokenize import sentence_tokenize, indic_tokenize

# Load Hindi Model in spaCy
spacy_udpipe.download("hi")  # Download once if not installed
nlp = spacy_udpipe.load("hi")  # Load the Hindi UDPipe model

# Tokenization using `indic-nlp-library`
def tokenize_indic(text, lang="hi"):
    sentences = sentence_tokenize.sentence_split(text, lang)
    words = [indic_tokenize.trivial_tokenize(sent) for sent in sentences]
    return sentences, words
    
# Function to get POS & NER

def process_text(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]  # POS tagging
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]  # Named Entities
    return pos_tags, named_entities

# Example Hindi Text

text = "क्रिप्टोग्राफी एक महत्वपूर्ण विषय है। यह सूचना सुरक्षा में उपयोगी है।"
  
# Tokenization

sentences, words = tokenize_indic(text)

# POS Tagging & NER

pos_tags, named_entities = process_text(text)

# Print Results

print("\n🔹 Sentence Tokenization:")
for sent in sentences:
    print(f"- {sent}")

print("\n🔹 Word Tokenization:")

for i, word_list in enumerate(words):
    print(f"Sentence {i+1}: {word_list}")

print("\n🔹 POS Tagging:")

for word, pos in pos_tags:
    print(f"{word}: {pos}")

print("\n🔹 Named Entity Recognition (NER):")

if named_entities:
    for entity, label in named_entities:
        print(f"{entity}: {label}")
else:
    print("No named entities found.")