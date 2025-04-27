import json
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def cluster_with_lsa_hdbscan(papers):
    # Extract titles and abstracts
    texts = [
        f"{paper['title']}. {paper['abstract']}"
        for author_papers in papers.values()
        for paper in author_papers if paper.get('title') and paper.get('abstract')
    ]

    # Step 1: Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=20000, max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Step 2: Dimensionality Reduction using LSA (Truncated SVD)

    n_components = 200  # Number of dimensions to reduce to
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    explained_variance = np.cumsum(lsa.explained_variance_ratio_)
    print(f"Explained Variance: {explained_variance}")

    # Step 3: Apply HDBSCAN for clustering
    hdbscan_model = HDBSCAN(min_cluster_size=7, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    clusters = hdbscan_model.fit_predict(lsa_matrix)

    # Step 4: Analyze and Visualize Results
    visualize_lsa_clusters(lsa_matrix, clusters)

    # Create a mapping of clusters to texts
    cluster_to_texts = {}
    for idx, cluster in enumerate(clusters):
        if cluster not in cluster_to_texts:
            cluster_to_texts[cluster] = []
        cluster_to_texts[cluster].append(texts[idx])

    # Print cluster information
    for cluster, texts_in_cluster in cluster_to_texts.items():
        print(f"Cluster {cluster}:")
        for text in texts_in_cluster[:3]:  # Show the first 3 texts as examples
            print(f" - {text[:100]}...")  # Display the first 100 characters
        print(f"Total texts in cluster: {len(texts_in_cluster)}\n")

    ## Run topic coherence comparison
    cluster_to_keywords = {}
    tfidf_feature_names = vectorizer.get_feature_names_out()

    for cluster in set(clusters):
        if cluster != -1:
            cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster]
            tfidf_matrix_cluster = vectorizer.fit_transform(cluster_texts)
            cluster_keywords = [
                tfidf_feature_names[i] for i in np.argsort(tfidf_matrix_cluster.toarray().mean(axis=0))[-10:]
            ]
            cluster_to_keywords[cluster] = cluster_keywords

    # Print keywords per cluster
    for cluster, keywords in cluster_to_keywords.items():
        print(f"Cluster {cluster} Keywords: {keywords}") 
     

    return clusters, cluster_to_texts


def visualize_lsa_clusters(lsa_matrix, clusters):
    """
    Visualize the clusters on the first two dimensions of the LSA-reduced matrix.
    """
    x = lsa_matrix[:, 0]  # First LSA component
    y = lsa_matrix[:, 1]  # Second LSA component

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(x, y, c=clusters, cmap='tab10', s=50, alpha=0.7)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("LSA + HDBSCAN Clustering Visualization")
    plt.xlabel("LSA Component 1")
    plt.ylabel("LSA Component 2")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    author_dir = 'preprocessed_data'
    data = {}
    for author_file in os.listdir(author_dir):
        with open(os.path.join(author_dir, author_file), "r", encoding="utf-8") as file:
            data[author_file] = json.load(file)


    print(data)

    # Perform clustering
    # clusters, cluster_to_texts = cluster_with_lsa_hdbscan(data)
