import os

from data.create_embeddings import generate_embeddings, test_embeddings
from data.preprocess_data_main import preprocess_hindi_corpus
from models.summarize_csebuetnlp_mT5 import summarize_corpus
from data.build_graph import build_knowledge_graph


# RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv"
PROCESSED_OUTPUT_PATH = "data/processed_corpus.csv"
SUMMARY_OUTPUT_PATH = "src/models/summarize_csebuetnlp_mT5.py"


# Get project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define absolute paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/monolingual-n/raw_IITB.csv")
PROCESSED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed")
TOKENIZED_INPUT = os.path.join(PROJECT_ROOT, "data/processed/tokenized_IITB.csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_IITB.csv")
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")


SUMMARY_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("RAW_DATA_PATH:", RAW_DATA_PATH)  # Debugging
print("PROCESSED_OUTPUT_PATH:", PROCESSED_OUTPUT_PATH)  # Debugging


print("Running preprocessing")
# preprocess_hindi_corpus(RAW_DATA_PATH, max_lines=10000)

print("Running summarization")
# summarize_corpus(
#     input_path=TOKENIZED_INPUT,
#     output_path=SUMMARY_PATH,
#     chunk_size=5,
#     max_lines=10000
# )
print("Summarization Completed")

# generate_embeddings(SUMMARY_PATH, EMBEDDING_PATH, SUMMARY_MODEL_NAME)
# test_embeddings(EMBEDDING_PATH)

# Call graph builder
build_knowledge_graph(
    summary_path=SUMMARY_PATH,
    embedding_path=EMBEDDING_PATH,
    graph_path=GRAPH_PATH,
    max_rows=10000,  # Optional
    similarity_threshold=0.5  # Optional
)