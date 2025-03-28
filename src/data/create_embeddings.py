from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
# import faiss

# Load preprocessed corpus CSV
corpus_file = "summarized_IITB.csv"  # Your processed corpus file
df = pd.read_csv(corpus_file, nrows=100)

# Load the embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for each sentence
sentences = df.iloc[:, 0].tolist()  # Assuming text is in the first column
embeddings = model.encode(sentences, convert_to_numpy=True)

# Save embeddings for later use
np.save("hindi_embeddings.npy", embeddings)

print(f"Generated embeddings shape: {embeddings.shape}")
