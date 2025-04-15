import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from .build_faiss_index import build_faiss_index
from .query_expansion import expand_query_with_fasttext

def load_index_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def get_query_embedding(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embedding = model.encode([query])
    return embedding.astype("float32")

def search_index(index, embedding, metadata, top_k=5):
    distances, indices = index.search(embedding, top_k)
    results = metadata.iloc[indices[0]]
    results["score"] = distances[0]
    return results

def retrieve_top_k_results(query, top_k=5):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph", "community_faiss.index")
    META_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph", "community_meta.pkl")

    build_faiss_index(index_path=INDEX_PATH, meta_path=META_PATH)
    
    index, metadata = load_index_and_metadata(INDEX_PATH, META_PATH)
    
    fasttext_path = os.path.join(PROJECT_ROOT, "models", "cc.hi.300.bin")
    expanded_query = expand_query_with_fasttext(query, model_path=fasttext_path)
    print(f"Expanded Query: {expanded_query}")
    query_embedding = get_query_embedding(expanded_query)
    return search_index(index, query_embedding, metadata, top_k=top_k)


if __name__ == "__main__":
    query = input("🔍 Enter your query: ")
    top_results = retrieve_top_k_results(query)

    print("\n🎯 Top Matching Community Summaries:\n")
    for idx, row in top_results.iterrows():
        print(f"Community: {idx} | Score: {row['score']:.4f}")
        print(f"Summary: {row['summary']}\n")



