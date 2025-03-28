# import re

# def detect_english_words(text):
#     """Finds all English words in the given text."""
#     return re.findall(r"\b[a-zA-Z]+\b", text)  # Matches only English words

# def count_english_words_in_file(file_path):
#     """Counts all English words in a text file (corpus)."""
#     english_word_count = 0
    
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             english_words = detect_english_words(line)  # Extract English words
#             english_word_count += len(english_words)  # Count them
            
#     return english_word_count

# # Example: Count English words in a corpus file
# corpus_file = "cleaned_output.txt"  # Change this to your actual file path
# total_english_words = count_english_words_in_file(corpus_file)

# print(f"Total English words in the corpus: {total_english_words}")


# def print_first_n_lines(file_path, n=1000):
#     """Prints the first `n` lines of a text file."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         for i, line in enumerate(file):
#             if i >= n:  # Stop after printing n lines
#                 break
#             print(line.strip())  # Print line without extra newlines

# # Example usage
# file_path = "output.txt"  # Change this to your actual file
# print_first_n_lines(file_path)


import numpy as np

# Load the saved embeddings
embeddings = np.load("hindi_embeddings.npy")
from sklearn.metrics.pairwise import cosine_similarity

# Load sample sentences
sent1_emb = embeddings[3].reshape(1, -1)
sent2_emb = embeddings[2].reshape(1, -1)

# Compute cosine similarity
similarity = cosine_similarity(sent1_emb, sent2_emb)[0][0]
print(f"Similarity between sentence 1 & 2: {similarity:.4f}")

