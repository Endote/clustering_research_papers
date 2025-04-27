from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import random
import pandas as pd
import csv


def read_authors_csv(file_path):
    """Read author names from a structured CSV file.
    
    Expects a header row with a column labeled 'Name'.
    """
    authors = []
    with open(file_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract the author name if the key "Name" exists.
            if "Name" in row and row["Name"].strip():
                authors.append(row["Name"].strip())
    return authors

def hopkins_statistic(X, sample_size=0.1):
    """Calculates the Hopkins statistic for a dataset."""
    if isinstance(sample_size, float):
        sample_size = int(sample_size * X.shape[0])

    random_indices = random.sample(range(X.shape[0]), sample_size)
    random_samples = X[random_indices, :]

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    uniform_samples = np.random.uniform(low=min_vals, high=max_vals, size=(sample_size, X.shape[1]))

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    distances_real, _ = nn.kneighbors(random_samples)
    distances_synthetic, _ = nn.kneighbors(uniform_samples)

    W = np.sum(distances_real[:, 1])
    U = np.sum(distances_synthetic[:, 1])

    return W / (W + U)


def get_top_terms_per_cluster(tfidf_matrix, labels, vectorizer, top_n=10):
    """Identify top terms per cluster."""
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}

    for cluster in np.unique(labels):
        # if cluster == -1:  # Skip noise points
            # continue
        cluster_data = tfidf_matrix[labels == cluster]
        mean_tfidf = np.asarray(cluster_data.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        cluster_terms[cluster] = [terms[i] for i in top_indices]
    return cluster_terms


def compute_coherence(tfidf_matrix, clusters, top_terms, vectorizer):
    """Compute coherence score for clusters."""
    coherence_scores = {}
    terms = vectorizer.get_feature_names_out()

    for cluster_id, cluster_docs in clusters.items():
        cluster_matrix = tfidf_matrix[cluster_docs, :]
        avg_weights = cluster_matrix.mean(axis=0).A1
        top_indices = [terms.tolist().index(term) for term in top_terms[cluster_id] if term in terms]
        coherence_scores[cluster_id] = avg_weights[top_indices].mean() if top_indices else 0
    return coherence_scores


def rank_clusters_by_coherence(tfidf_matrix, labels, vectorizer):
    """Rank clusters by coherence scores."""
    clusters = {label: np.where(labels == label)[0].tolist() for label in np.unique(labels)}
    top_terms = get_top_terms_per_cluster(tfidf_matrix, labels, vectorizer)
    coherence_scores = compute_coherence(tfidf_matrix, clusters, top_terms, vectorizer)

    sorted_scores = sorted(coherence_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_clusters = {k: clusters[k] for k, _ in sorted_scores}
    sorted_top_terms = {k: top_terms[k] for k, _ in sorted_scores}

    return sorted_clusters, sorted_top_terms, sorted_scores


def cluster_with_hdbscan(papers, n_neighbors=15, min_dist=0.1, n_components=200, min_cluster_size=15, min_samples=5):
    """Cluster using UMAP and HDBSCAN."""
    texts = [
        paper['abstract']
        for author_papers in papers.values()
        for paper in author_papers if paper.get('abstract')
    ]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=0.02, max_df=0.8, max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # UMAP dimensionality reduction
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    reduced_data = umap.fit_transform(tfidf_matrix.toarray())

    # No dimensionality reduction
    # reduced_data = tfidf_matrix.toarray()

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(reduced_data)

    if len(set(labels)) <= 1:
        print("\nNo valid clusters found. Try adjusting HDBSCAN or UMAP parameters.")
        return None, tfidf_matrix, vectorizer, reduced_data

    # Silhouette score (ignoring noise points)
    filtered_data = reduced_data
    filtered_labels = labels
    silhouette = silhouette_score(filtered_data, filtered_labels) if len(set(filtered_labels)) > 1 else -1
    print(f"\nSilhouette Score: {silhouette:.4f}")

    return labels, tfidf_matrix, vectorizer, reduced_data

def load_author_list(filepath):
    """Load and preprocess the author list from the specified file."""
    with open(filepath, "r", encoding="utf-8") as file:
        authors = [line.strip() for line in file.readlines()]
    return set(authors)

def get_clusters_per_author(data, labels, valid_authors):
    """Aggregate clusters and their paper counts per author."""
    clusters_per_author = {}

    # Flatten the data to map each abstract to its authors
    abstracts_authors = [
        (paper['abstract'], paper['authors'])
        for author_papers in data.values()
        for paper in author_papers if paper.get('abstract') and paper.get('authors')
    ]

    # Associate each abstract with its cluster
    for i, (abstract, authors) in enumerate(abstracts_authors):
        cluster_id = labels[i]
        for author in authors.split(", "):  # Split multiple authors by comma
            # if author not in valid_authors:
            #     continue
            if author not in clusters_per_author:
                clusters_per_author[author] = {}
            if cluster_id not in clusters_per_author[author]:
                clusters_per_author[author][cluster_id] = 0
            clusters_per_author[author][cluster_id] += 1

    return clusters_per_author

import unicodedata
from collections import Counter
import pandas as pd

# Updated version of the export function with all normalization features and detailed prints

def normalize_author_name_verbose(name: str) -> str:
    print(f"Original name: {name}")
    name = name.strip()
    parts = name.split()
    if len(parts) < 2:
        print(f"⚠️ Skipping malformed name: {name}")
        return name

    # Check if reversed name is likely (surname first)
    is_reversed = parts[0].isupper() or parts[1][0].islower()
    if is_reversed:
        parts = parts[::-1]
        print(f"🔄 Detected reversed format, reordering: {' '.join(parts)}")

    first = parts[0].capitalize()
    last = " ".join(parts[1:]).capitalize()

    # Remove accents
    first_no_accents = unicodedata.normalize('NFKD', first).encode('ASCII', 'ignore').decode()
    last_no_accents = unicodedata.normalize('NFKD', last).encode('ASCII', 'ignore').decode()
    normalized_name = f"{first_no_accents} {last_no_accents}"
    print(f"✅ Normalized to: {normalized_name}")
    return normalized_name

def export_semantic_nodes_verbose(clusters_per_author, sorted_top_terms, output_path="semantic_nodes.csv"):
    """
    Exports a semantic node table with dominant clusters and top terms per author.
    Normalizes author names to canonical form with verbose output.
    """
    records = []
    for author, cluster_counts in clusters_per_author.items():
        norm_author = normalize_author_name_verbose(author)
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
    pre_dedup = len(semantic_df)
    semantic_df.drop_duplicates(subset=["Id"], keep="first", inplace=True)
    post_dedup = len(semantic_df)

    semantic_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Exported semantic node info to: {output_path}")
    print(f"🧹 Deduplicated {pre_dedup - post_dedup} records (from {pre_dedup} to {post_dedup})")


if __name__ == '__main__':
    author_dir = 'preprocessed_data_new'
    data = {}
    for author_file in os.listdir(author_dir):
        with open(os.path.join(author_dir, author_file), "r", encoding="utf-8") as file:
            data[author_file] = json.load(file)

    # Perform clustering with HDBSCAN
    labels, tfidf_matrix, vectorizer, reduced_data = cluster_with_hdbscan(data, n_neighbors=12, min_dist=0.05, n_components=400, min_cluster_size=12, min_samples=5)

    # Load valid authors from the list
    # valid_authors = load_author_list("list.txt")
    valid_authors = read_authors_csv('data/faculty_profiles.csv')

    if labels is not None:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(reduced_data)
        plt.figure(figsize=(10, 7))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis', s=5)
        plt.colorbar()
        plt.title(f"Cluster Visualization (UMAP + HDBSCAN)")
        plt.show()

        # Rank clusters by coherence
        sorted_clusters, sorted_top_terms, sorted_scores = rank_clusters_by_coherence(tfidf_matrix, labels, vectorizer)
        print(str(len(sorted_clusters)) + " Clusters (sorted by coherence scores):")
        for cluster_id, paper_indices in sorted_clusters.items():
            print(f"Cluster {cluster_id} - count {len(paper_indices)}: {paper_indices}")

        print("\nTop Terms per Cluster (sorted by coherence scores):")
        for cluster_id, terms in sorted_top_terms.items():
            print(f"Cluster {cluster_id}: {', '.join(terms)}")

        print("\nCoherence Scores (sorted):")
        for cluster_id, score in sorted_scores:
            print(f"Cluster {cluster_id}: {score:.4f}")

        hopkins = hopkins_statistic(reduced_data)
        print(f"\nHopkins Statistic: {hopkins:.4f}")

        # Get clusters per author (filtered by valid authors)
        clusters_per_author = get_clusters_per_author(data, labels, valid_authors)

        print("\nClusters and Paper Counts for the Authors:")
        for author, clusters in clusters_per_author.items():
            print(f"{author}:")
            for cluster_id, count in sorted(clusters.items()):
                cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                print(f"  {cluster_name}: {count} papers")


        export_semantic_nodes_verbose(clusters_per_author, sorted_top_terms)

        print("\n✅ Exported 'semantic_nodes.csv' with enriched author info for Gephi.")

