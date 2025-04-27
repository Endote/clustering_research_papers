import os
import json
import pandas as pd
import unicodedata
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import random
from unidecode import unidecode


# --- Normalization ---
def normalize_author_name(name: str) -> str:
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    if parts[0].isupper() or parts[1][0].islower():
        parts = parts[::-1]
    first = unidecode(parts[0].capitalize())
    last = unidecode(" ".join(parts[1:]).capitalize())
    return f"{first} {last}"


# --- Data Collection ---
def load_json_data(data_dir):
    data = {}
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                data[file] = json.load(f)
    return data

# --- Clustering ---
def cluster_with_hdbscan(papers):
    texts = [p['abstract'] for papers_list in papers.values() for p in papers_list if p.get('abstract')]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=0.02, max_df=0.8, max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    reducer = UMAP(n_neighbors=12, min_dist=0.05, n_components=400)
    reduced = reducer.fit_transform(tfidf_matrix.toarray())

    clusterer = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=5)
    labels = clusterer.fit_predict(reduced)
    print(f"Silhouette score: {silhouette_score(reduced, labels):.4f}")
    return labels, tfidf_matrix, vectorizer

def get_clusters_per_author(data, labels):
    abstracts_authors = [(p['abstract'], p['authors']) for papers_list in data.values() for p in papers_list if p.get('abstract') and p.get('authors')]
    author_clusters = defaultdict(lambda: defaultdict(int))
    for i, (_, authors) in enumerate(abstracts_authors):
        for author in authors.split(", "):
            author = normalize_author_name(author)
            author_clusters[author][labels[i]] += 1
    return author_clusters

def get_top_terms(tfidf_matrix, labels, vectorizer, top_n=10):
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for c in np.unique(labels):
        cluster_data = tfidf_matrix[labels == c]
        avg = np.asarray(cluster_data.mean(axis=0)).flatten()
        top_terms = [terms[i] for i in avg.argsort()[-top_n:][::-1]]
        cluster_terms[c] = top_terms
    return cluster_terms

# --- Export Semantic Node Table ---
def export_semantic_nodes(clusters_per_author, sorted_top_terms, output_path="semantic_nodes.csv"):
    records = []
    for author, cluster_counts in clusters_per_author.items():
        norm_author = normalize_author_name(author)
        total = sum(cluster_counts.values())
        most_common_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]
        top_terms = sorted_top_terms.get(most_common_cluster, []) if most_common_cluster != -1 else []
        records.append({
            "Id": norm_author,
            "Label": author,
            "SemanticCluster": f"Cluster {most_common_cluster}" if most_common_cluster != -1 else "Noise",
            "ClusterTerms": ", ".join(top_terms),
            "TotalPapers": total,
            "TopClusterCount": cluster_counts[most_common_cluster]
        })

    semantic_df = pd.DataFrame(records)
    pre = len(semantic_df)
    semantic_df.drop_duplicates(subset=["Id"], keep="first", inplace=True)
    post = len(semantic_df)
    print(f"🧹 Deduplicated {pre - post} authors (from {pre} to {post})")
    semantic_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported semantic node info to: {output_path}")


# --- Main ---
if __name__ == "__main__":
    data = load_json_data("preprocessed_data_new")
    labels, tfidf_matrix, vectorizer = cluster_with_hdbscan(data)
    author_clusters = get_clusters_per_author(data, labels)
    top_terms = get_top_terms(tfidf_matrix, labels, vectorizer)
    export_semantic_nodes(author_clusters, top_terms)
