import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities




def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    # pos = nx.spring_layout(G, seed=42, k=0.4) #layout1
    pos = nx.kamada_kawai_layout(G) #layout2

    # Get communities and assign colors
    communities = set(nx.get_node_attributes(G, "community").values())
    color_map = cm.get_cmap("tab20", len(communities))  # distinct colors
    node_colors = [color_map(G.nodes[node]["community"]) for node in G.nodes()]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [1 + 4 * w for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, "weight")
    formatted_labels = {(u, v): f"{d:.2f}" for (u, v), d in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_size=8)

    plt.title("📘 Summary Similarity Graph (Colored by Community)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def build_knowledge_graph(summary_path="summarized_IITB.csv",
                          embedding_path="summarized_embeddings.npy",
                          graph_path="summary_graph.graphml",
                          max_rows=None,
                          similarity_threshold=0.5):


    print(f"Loading summarized data from: {summary_path}")
    df = pd.read_csv(summary_path, nrows=max_rows)

    print(f"Loading embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path)

    #sanity check - if embedding and summary idx matches
    assert len(df) == len(embeddings), "Mismatch between number of summaries and embeddings!"

    # Initialize graph and add nodes
    G = nx.Graph()
    #Storing idx and summary as nodes and not embeddings as GraphML file doesn't support high-dimensional arrays well
    #also we can access embedding using the idx of the node. The i-th row in the summary CSV → is related to the i-th embedding in the .npy array.
    for idx, row in df.iterrows():
        G.add_node(idx, text=row["summary"]) 

    similarity_matrix = cosine_similarity(embeddings)
    print("\n🔍 Sample Similarity Scores:")
    num_nodes = len(embeddings)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            print(f"Similarity between {i} and {j}: {similarity_matrix[i][j]:.4f}")
            

    # Add edges based on threshold
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = similarity_matrix[i][j] #similarity_matrix[0][1] corresponds to similarity between node 0 and node 1
            if score >= similarity_threshold:
                G.add_edge(i, j, weight=score)

    print(f"🔗 Total edges added (threshold ≥ {similarity_threshold}): {G.number_of_edges()}")

    communities = list(greedy_modularity_communities(G))
    for i, community in enumerate(communities):
        for node in community:
            G.nodes[node]["community"] = i
            
    # Save graph
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    nx.write_graphml(G, graph_path)
    print(f"Knowledge Graph saved to: {graph_path}")

    community_labels_path = os.path.join(os.path.dirname(graph_path), "community_labels.csv")
    os.makedirs(os.path.dirname(community_labels_path), exist_ok=True)
    community_map = {node: G.nodes[node]['community'] for node in G.nodes()}
    pd.DataFrame.from_dict(community_map, orient='index', columns=["community"]).to_csv(community_labels_path)
    print(f"Community labels saved to: {community_labels_path}")

    print("Visualising graph:")
    visualize_graph(G)

    print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")
    for u, v, data in list(G.edges(data=True))[:5]:
        print(f"🔗 {u} -- {v} (score: {data['weight']:.2f})")

