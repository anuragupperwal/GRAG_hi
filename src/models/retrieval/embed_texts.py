import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_summaries(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["summary"].fillna("").tolist()
    return texts, df

def get_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings