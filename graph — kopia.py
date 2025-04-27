import os
import json
import pandas as pd
from collections import defaultdict
from itertools import combinations
import unicodedata
from unidecode import unidecode
import networkx as nx

try:
    import community as community_louvain
except ImportError:
    raise ImportError("Please install the `python-louvain` package with `pip install python-louvain`")


def normalize_author_name(name: str) -> str:
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    if parts[0].isupper() or parts[1][0].islower():
        parts = parts[::-1]
    first = unidecode(parts[0].capitalize())
    last = unidecode(" ".join(parts[1:]).capitalize())
    return f"{first} {last}"


def build_coauthorship_graph(data_dir="preprocessed_data_new"):
    coauthor_counts = defaultdict(int)
    authors_all = set()

    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                papers = json.load(f)
                for paper in papers:
                    if "authors" in paper and paper["authors"]:
                        authors = [normalize_author_name(a.strip()) for a in paper["authors"].split(",") if a.strip()]
                        authors_all.update(authors)
                        for a1, a2 in combinations(sorted(authors), 2):
                            coauthor_counts[(a1, a2)] += 1

    sorted_authors = sorted(authors_all)
    adj_df = pd.DataFrame(0, index=sorted_authors, columns=sorted_authors)

    for (a1, a2), count in coauthor_counts.items():
        adj_df.loc[a1, a2] = count
        adj_df.loc[a2, a1] = count

    adj_df.to_csv("coauthorship_matrix_raw.csv", encoding="utf-8-sig")
    print("✅ Saved coauthorship matrix as 'coauthorship_matrix_raw.csv'")

    edges = []
    for i, a1 in enumerate(adj_df.index):
        for j, a2 in enumerate(adj_df.columns):
            if j <= i: continue
            weight = adj_df.iloc[i, j]
            if weight > 0:
                edges.append((a1, a2, weight))

    edge_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])
    edge_df.to_csv("gephi_edge_list.csv", index=False, encoding="utf-8-sig")
    print("✅ Saved Gephi edge list as 'gephi_edge_list.csv'")

    return "gephi_edge_list.csv"


def detect_and_export_communities(edge_list_csv: str, output_csv: str = "detected_communities.csv"):
    edges_df = pd.read_csv(edge_list_csv)
    G = nx.Graph()

    for _, row in edges_df.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

    print(f"🔍 Detecting communities with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")
    partition = community_louvain.best_partition(G, weight="weight")
    community_df = pd.DataFrame(partition.items(), columns=["Author", "Community"])
    community_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Louvain community labels saved to: {output_csv}")
    return community_df


def cross_reference_communities_with_semantic_clusters(communities_csv="detected_communities.csv", semantic_csv="semantic_nodes.csv", output_csv="community_semantic_topics.csv"):
    communities_df = pd.read_csv(communities_csv)
    semantic_df = pd.read_csv(semantic_csv)

    merged = pd.merge(communities_df, semantic_df, how="left", left_on="Author", right_on="Id")
    merged.to_csv("merged_communities_with_semantics.csv", index=False, encoding="utf-8-sig")

    grouped = (
        merged.groupby("Community")["SemanticCluster"]
        .value_counts()
        .groupby(level=0)
        .nlargest(3)
        .reset_index(level=0)
        .rename(columns={"SemanticCluster": "TopSemanticTopics"})
    )

    grouped.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Cross-referenced semantic topics per community saved to: {output_csv}")


if __name__ == "__main__":
    edge_list_file = build_coauthorship_graph()
    community_df = detect_and_export_communities(edge_list_file)
    cross_reference_communities_with_semantic_clusters()
