import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities
from collections import defaultdict

from data.community_summarization import summarize_communities


def visualize_with_plotly(G):
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=[G.nodes[n].get('community', 0) for n in G.nodes()],
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Knowledge Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)))
    
    fig.show()



def build_knowledge_graph(summary_path="summarized_IITB.csv",
                          embedding_path="summarized_embeddings.npy",
                          graph_path="summary_graph.graphml",
                          max_rows=None,
                          similarity_threshold=0.5):


    print(f"Loading summarized data from: {summary_path}")
    df = pd.read_csv(summary_path, nrows=max_rows)
    print(f"Loading embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path)

    if max_rows is not None:
        embeddings = embeddings[:len(df)]

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

    #Louvain's community detection algorithm
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
    # visualize_graph(G)
    visualize_with_plotly(G)

    print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")
    for u, v, data in list(G.edges(data=True))[:5]:
        print(f"{u} -- {v} (score: {data['weight']:.2f})")


    # #Check graph
    # # Group node summaries by community
    community_groups = defaultdict(list)
    for node, community_id in community_map.items():
        community_groups[community_id].append(G.nodes[node]["text"])


    # Show 2–3 sample summaries per community
    print("\n🧠 Sample summaries per community:\n")
    summary_records = []
    for cid, summaries in community_groups.items():
        print(f"Community {cid} (Total {len(summaries)} summaries):")
        for summary in summaries:  # show first 3
            print("  🔹", summary.strip())
            summary_records.append({"Community": cid, "Summary": summary.strip()})
        print()
            # Save all shown summaries to CSV
    test_path = os.path.join(os.path.dirname(graph_path), "per_communities_summaries_test.csv")
    pd.DataFrame(summary_records).to_csv(test_path, index=False)
    print(f"\n📝 Saved sample summaries per community to: {test_path}")


    #generate community summaries
    summarize_communities(G, output_path=os.path.dirname(graph_path))



if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_IITB.csv")
    EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")
    GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")

    build_knowledge_graph(summary_path=SUMMARY_PATH, embedding_path=EMBEDDING_PATH, graph_path=GRAPH_PATH, max_rows=1000, similarity_threshold=0.5)
    