from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

def compute_coherence(tfidf_matrix, clusters, top_terms, vectorizer):
    """Computes coherence score for clusters."""
    coherence_scores = {}
    terms = vectorizer.get_feature_names_out()
    for cluster_id, cluster_docs in clusters.items():
        cluster_matrix = tfidf_matrix[cluster_docs, :]
        avg_weights = cluster_matrix.mean(axis=0).A1  # Average TF-IDF weights
        top_indices = [terms.tolist().index(term) for term in top_terms[cluster_id] if term in terms]
        coherence_scores[cluster_id] = avg_weights[top_indices].mean() if top_indices else 0
    return coherence_scores

def get_top_terms_per_cluster(tfidf_matrix, labels, vectorizer, top_n=10):
    terms = vectorizer.get_feature_names_out()
    clusters = pd.DataFrame(tfidf_matrix.toarray())
    cluster_terms = {}
    for cluster in np.unique(labels):
        cluster_center = clusters[labels == cluster].mean(axis=0)
        top_terms = [terms[i] for i in cluster_center.argsort()[-top_n:][::-1]]
        cluster_terms[cluster] = top_terms
    return cluster_terms

def plot_dendrogram(model, **kwargs):
    """Plots the dendrogram for hierarchical clustering."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def hopkins_statistic(X, sample_size=0.1):
    """Calculates the Hopkins statistic for a dataset."""
    if isinstance(sample_size, float):
        sample_size = int(sample_size * X.shape[0])

    # Randomly select samples from the dataset
    random_indices = random.sample(range(X.shape[0]), sample_size)
    random_samples = X[random_indices, :]

    # Generate synthetic uniform data
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    uniform_samples = np.random.uniform(low=min_vals, high=max_vals, size=(sample_size, X.shape[1]))

    # Calculate distances
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    distances_real, _ = nn.kneighbors(random_samples)
    distances_synthetic, _ = nn.kneighbors(uniform_samples)

    W = np.sum(distances_real[:, 1])
    U = np.sum(distances_synthetic[:, 1])

    return W / (W + U)

def cluster(papers, n_components=50, metric='euclidean', linkage='ward', max_clusters=None, distance_threshold=1):
    """
    Perform clustering on a set of papers and return clusters and their top terms.

    Parameters:
        papers (dict): Dictionary of papers with abstracts.
        n_components (int): Number of components for PCA.
        metric (str): Metric used to compute the linkage ('euclidean', 'cosine', etc.).
        linkage (str): Linkage criterion ('ward', 'complete', 'average', etc.).
        max_clusters (int or None): Maximum number of clusters (None for no pre-defined number).
        distance_threshold (float): Threshold for forming clusters when max_clusters is None.

    Returns:
        dict: Clusters with paper indices.
        dict: Top terms for each cluster.
    """
    # 1. Load data
    texts = [
        paper['abstract']
        for author_papers in papers.values()
        for paper in author_papers if paper.get('abstract')
    ]

    # 2. Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.8, max_features=12000, stop_words='english', )#ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 3. Dimensionality reduction
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    # reduced_data =tfidf_matrix.toarray()

    # 4. Clustering
    if max_clusters is None:
        clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric=metric,
            linkage=linkage,
            compute_full_tree=True
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=max_clusters,
            metric=metric,
            linkage=linkage,
            compute_full_tree=True
        )

    labels = clustering.fit_predict(reduced_data)

    # 5. Analyze clusters
    # Group papers by clusters
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    # Get top terms for each cluster
    top_terms = get_top_terms_per_cluster(tfidf_matrix, labels, vectorizer)

    # 6. Visualize clusters
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(reduced_data)
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("Cluster Visualization")
    plt.show()

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    plot_dendrogram(clustering, truncate_mode='level', p=5)
    plt.title("Dendrogram")
    plt.show()

    return clusters, top_terms, tfidf_matrix, labels, vectorizer, reduced_data

if __name__ == '__main__':
    author_dir = 'preprocessed_data'
    data = {}
    for author_file in os.listdir(author_dir):
        with open(os.path.join(author_dir, author_file), "r", encoding="utf-8") as file:
            data[author_file] = json.load(file)

    # Perform clustering
    clusters, top_terms, tfidf_matrix, labels, vectorizer, reduced_data = cluster(
        data,
        n_components=50,
        metric='cosine',
        linkage='average', # 'ward', 'complete', 'average', 'single'
        max_clusters=None,
        distance_threshold=0.3
    )

    # Compute coherence scores
    coherence_scores = compute_coherence(tfidf_matrix, clusters, top_terms, vectorizer)

    # Sort clusters by coherence scores
    sorted_scores = sorted(coherence_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_clusters = {k: clusters[k] for k, _ in sorted_scores}
    sorted_top_terms = {k: top_terms[k] for k, _ in sorted_scores}

    # Print clusters and their respective paper indices
    print("Clusters (sorted by coherence scores):")
    for cluster_id, paper_indices in sorted_clusters.items():
        print(f"Cluster {cluster_id} - count {len(paper_indices)}: {paper_indices}")

    # Print top terms for each cluster
    print("\nTop Terms per Cluster (sorted by coherence scores):")
    for cluster_id, terms in sorted_top_terms.items():
        print(f"Cluster {cluster_id}: {', '.join(terms)}")

    # Print coherence scores
    print("\nCoherence Scores (sorted):")
    for cluster_id, score in sorted_scores:
        print(f"Cluster {cluster_id}: {score:.4f}")

    # Compute silhouette score
    silhouette = silhouette_score(tfidf_matrix.toarray(), labels)
    print(f"\nSilhouette Score: {silhouette:.4f}")

    # Compute Hopkins statistic
    # hopkins = hopkins_statistic(reduced_data, 1)
    hopkins = hopkins_statistic(tfidf_matrix.toarray(), sample_size=1)
    print(f"\nHopkins Statistic: {hopkins:.4f}")
