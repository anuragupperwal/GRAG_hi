import os
import faiss
import numpy as np
import pickle
from .embed_texts import load_summaries, get_embeddings


def save_metadata(df, output_meta_path):
    df.to_pickle(output_meta_path)

def build_faiss_index(index_path, meta_path):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    SUMMARY_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph", "community_summary.csv")
    # INDEX_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph", "community_faiss.index")
    # METADATA_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph", "community_meta.pkl")

    texts, df = load_summaries(SUMMARY_CSV_PATH)
    embeddings = get_embeddings(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    save_metadata(df, meta_path)

    print(f"FAISS index saved to {index_path}")
    print(f"Metadata saved to {meta_path}")




if __name__ == "__main__":
    build_faiss_index()
