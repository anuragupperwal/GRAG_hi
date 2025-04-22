import os
import networkx as nx

from data.create_embeddings import generate_embeddings, test_embeddings
from data.preprocess_data_main import preprocess_hindi_corpus
from models.summarize_csebuetnlp_mT5 import summarize_corpus
from data.build_graph import build_knowledge_graph
from data.community_summarization import summarize_communities
from data.retrieve_and_generate import generate_output


# RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv"
# PROCESSED_OUTPUT_PATH = "data/processed_corpus.csv"


# Get project root dynamically
PROJECT_ROOT = "/kaggle/working/"
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define absolute paths
RAW_DATA_PATH = "/kaggle/input/hindi-corpus/raw_IITB.csv"
TOKENIZED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_IITB.csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_IITB.csv")
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")
SUMMARY_GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/output/answer.txt")


max_line_bound = 1000
print("Running preprocessing")
preprocess_hindi_corpus(RAW_DATA_PATH, max_lines=max_line_bound, project_root=PROJECT_ROOT)

print("Running summarization")
summarize_corpus(
    input_path=TOKENIZED_PATH,
    output_path=SUMMARY_PATH,
    chunk_size=5,
    max_lines=max_line_bound
)
print("Summarization Completed")


generate_embeddings(SUMMARY_PATH, EMBEDDING_PATH, max_line_bound)
# test_embeddings(EMBEDDING_PATH)

print("Building Graph")
build_knowledge_graph(
    summary_path=SUMMARY_PATH,
    embedding_path=EMBEDDING_PATH,
    graph_path=GRAPH_PATH,
    top_k=5
)

print("Summarizing Communities")
G = nx.read_graphml(GRAPH_PATH)
summarize_communities(G, output_path_directory=SUMMARY_GRAPH_PATH)

# retrieve based on query
query1 = "भारत में चर्चित राजनीतिक और कानूनी मामलों का सामाजिक प्रभाव क्या रहा है?"
# Generated:  बीबीसी हिंदी सेवा के विशेष कार्यक्रम 'एक मुलाक़ात' में हम भारत के पूर्व प्रधानमंत्री लालकृष्ण आडवाणी और उनके सहयोगी रामलला के जन्मस्थान पर राम मंदिर बनाने के मामले को लेकर बात कर रहे थे..
# Generated:  जाति और जनगणना को जोड़ने की मांग को लेकर भारतीय जनता पार्टी सरकार के लिए आसान नहीं है क्योंकि इससे जुड़े कई सवालों का जवाब देने के बावजूद भाजपा सरकार की तरफ़ से इस्तीफ़ा देना पड़ रहा है.
query2 = "निम्नलिखित जानकारी को ध्यान में रखते हुए, भारत में राजनीतिक और कानूनी मामलों के सामाजिक प्रभाव का विश्लेषण करें"
# Generated:  भारत में राजनीतिक और कानूनी मामलों के सामाजिक प्रभाव का विश्लेषण
query3 = "भारत में अंतरिक्ष अनुसंधान के क्षेत्र में इसरो की भूमिका क्या है?"
# Generated:  भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) की भूमिका क्या है?

# query = input("Enter your query: ")
top_k = 5
print("in dataset: ")
generate_output(top_k, query1, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
print("in dataset 2 : ")
generate_output(top_k, query2, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)

print("Not in dataset: ")
generate_output(top_k, query3, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
