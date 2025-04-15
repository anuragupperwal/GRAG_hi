import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import networkx as nx


#Prevents the fork-related deadlock and slowness from huggingface tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"  #"ai4bharat/IndicT5-Summarization"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

def summarize_with_indicT5(texts, max_input_length=1024, max_output_length=256):
    """
    Summarize a list of Hindi texts using IndicT5
    """
    summaries = []
    for text in tqdm(texts, desc="Summarizing without IndicT5"):
        input_text = "summarize: " + text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_output_length,
                min_length=int(max_output_length * 0.7),
                length_penalty=0.8,
                num_beams=4,
                early_stopping=False
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f" Error during generation: {e}")
            summary = "[Error: Generation Failed]"
        
        summaries.append(summary)
    return summaries

def recursive_summarize(sentences, chunk_size=10, max_tokens=1000):
    """
    Recursively summarize sentences until a compact final summary is obtained.
    """
    total_tokens = sum(len(tokenizer.encode(s)) for s in sentences)
    current_sentences = sentences[:]
    while len(current_sentences) > 1:
        chunks = [" ".join(current_sentences[i:i+chunk_size]) for i in range(0, len(current_sentences), chunk_size)]
                
        # avg_input_length = sum(len(tokenizer.encode(s)) for s in chunks) // len(chunks)
        # max_output_length = int(avg_input_length * 1)  # 50% compression ratio
        # MIN_SUMMARY_LENGTH = 60
        # max_output_length = max(max_output_length, MIN_SUMMARY_LENGTH)   # minimum threshold to avoid empty summaries

        MIN_SUMMARY_LENGTH = 60
        compression_ratio = 0.6
        if total_tokens < MIN_SUMMARY_LENGTH:
            max_output_length = MIN_SUMMARY_LENGTH
        else:
            max_output_length = max(int(total_tokens * compression_ratio), MIN_SUMMARY_LENGTH)

        print(f"🧪 Total input tokens: {total_tokens} | Target summary tokens: {max_output_length}")

        # current_sentences = summarize_with_indicT5(chunks, max_output_length=max_output_length)


        #option 1.1
        # max_input_lengths = [min(len(tokenizer.encode(chunk)) + 20, 1024) for chunk in chunks]
        # current_sentences = [
        #     summarize_with_indicT5([chunk], max_input_length=length, max_output_length=max_output_length)[0]
        #     for chunk, length in zip(chunks, max_input_lengths)
        # ]

        # option 1.1.1
        current_sentences = []
        for chunk in chunks:
            input_len = min(len(tokenizer.encode(chunk)) + 20, 1024)
            summary = summarize_with_indicT5([chunk], max_input_length=input_len, max_output_length=max_output_length)[0]
            current_sentences.append(summary)

        # Optional exit condition
        token_counts = [len(tokenizer.encode(s)) for s in current_sentences]
        if sum(token_counts) <= max_tokens:
            break

    return current_sentences

def get_top_nodes(G, node_list, top_k_ratio=0.9):
    """
    Select top-k% nodes from a community using degree centrality
    """
    centrality = nx.degree_centrality(G.subgraph(node_list))
    top_k = int(len(node_list) * top_k_ratio)
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_k]
    return top_nodes

def summarize_communities(G, output_path_directory=None):
    from collections import defaultdict

    community_groups = defaultdict(list)
    for node in G.nodes():
        cid = G.nodes[node].get("community", -1)
        community_groups[cid].append(node)

    all_summaries = {}
    for community_id, node_list in community_groups.items():
        top_nodes = get_top_nodes(G, node_list)
        top_summaries = [G.nodes[n]['text'] for n in top_nodes]
        final_summary = recursive_summarize(top_summaries)
        summary_text = final_summary[0] if final_summary else "[No summary generated]"
        all_summaries[community_id] = summary_text

        print(f"\n Final summary for Community {community_id}:")
        print(summary_text)

    if output_path_directory:
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        community_summary_path = os.path.join(os.path.dirname(output_path_directory), "community_summary.csv")
        pd.DataFrame.from_dict(all_summaries, orient='index', columns=['summary']).to_csv(community_summary_path)
        print(f"Community summaries saved at: {community_summary_path}")

    return all_summaries




if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #to go to GRAG_hi
    GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/")
    G = nx.read_graphml(GRAPH_PATH)
    summarize_communities(G, output_path_directory=SUMMARY_PATH)