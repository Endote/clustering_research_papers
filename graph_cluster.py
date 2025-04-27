import json
import os
import numpy as np
import networkx as nx

from collections import Counter
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def calculate_cluster_coherence(cluster_to_texts):
    """
    Calculate coherence scores for each cluster using Gensim's CoherenceModel.
    Parameters:
    - cluster_to_texts: A dictionary where keys are cluster labels and values are lists of texts.
      Each text is a raw string, which we'll split into tokens.
    Returns:
    - A dictionary mapping cluster labels to their coherence scores.
    """
    coherence_scores = {}

    # 1. Tokenize all texts in each cluster
    tokenized_texts = {
        cluster: [text.split() for text in texts]
        for cluster, texts in cluster_to_texts.items()
    }

    # 2. Build a Gensim dictionary from *all* tokens across clusters
    #    (you could also build separate dictionaries per cluster if desired).
    all_docs = []
    for cluster, texts in tokenized_texts.items():
        all_docs.extend(texts)

    dictionary = Dictionary(all_docs)

    # 3. Loop over each cluster and compute coherence
    for cluster, texts in tokenized_texts.items():
        # Skip clusters with fewer than 5 texts
        if len(texts) < 5:
            print(f"Skipping cluster {cluster} due to insufficient data (size={len(texts)}).")
            continue

        # Skip clusters where all texts are empty
        if all(len(t) == 0 for t in texts):
            print(f"Skipping cluster {cluster} because it contains empty texts.")
            continue

        # Build the corpus for this cluster
        corpus = [dictionary.doc2bow(doc) for doc in texts]

        ### DEBUG

        # print(f"Cluster {cluster} => number of texts: {len(texts)}")
        # print(f"Dictionary size: {len(dictionary)}")
        # print(f"Sample doc2bow for first doc: {corpus[0] if corpus else 'None'}")

        # 4. Create and compute the coherence model
        try:
            # For each cluster, build a list of "top N words" that represent that cluster
            all_tokens = [token for doc in texts for token in doc]
            freq = Counter(all_tokens)
            # Pick, say, top 10 words
            top_words = [w for w, _ in freq.most_common(10)]
            print(top_words)

            # When building your CoherenceModel:
            coherence_model = CoherenceModel(
                topics=[top_words],   # <— a list of lists of words
                texts=texts,
                corpus=corpus,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_scores[cluster] = coherence_model.get_coherence()
        except Exception as e:
            print(f"Error calculating coherence for cluster {cluster}: {e}")
            coherence_scores[cluster] = float('nan')

    return coherence_scores

def graph_based_clustering(papers, embedding_model='all-MiniLM-L6-v2', dimensions=256, n_clusters=10):
    # Step 1: Extract texts
    texts = [
        f"{paper['title']}. {paper['abstract']}"
        for author_papers in papers.values()
        for paper in author_papers if paper.get('title') and paper.get('abstract')
    ]

    # Step 2: Generate Sentence Embeddings
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, show_progress_bar=True)

    # Step 3: Create a Similarity Graph
    similarity_matrix = cosine_similarity(embeddings)
    graph = nx.Graph()

    # Add nodes
    for i, text in enumerate(texts):
        graph.add_node(i, text=text)

    # Add edges with weights based on similarity
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            weight = similarity_matrix[i][j]
            if weight > 0.5:  # Threshold to filter weak edges
                graph.add_edge(i, j, weight=weight)

    # Step 4: Generate Node Embeddings
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=100, num_walks=500, workers=6)
    node_embeddings = node2vec.fit(window=10, min_count=1, batch_words=10)
    node_vectors = np.array([node_embeddings.wv[str(node)] for node in graph.nodes])


    # Step 5: Cluster Node Embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(node_vectors)

    # Step 6: Map Clusters to Texts
    cluster_to_texts = {}
    for idx, label in enumerate(cluster_labels):
        if label not in cluster_to_texts:
            cluster_to_texts[label] = []
        cluster_to_texts[label].append(graph.nodes[idx]['text'])

    # Print cluster information
    for cluster, texts_in_cluster in cluster_to_texts.items():
        print(f"Cluster {cluster}:")
        for text in texts_in_cluster[:3]:  # Show the first 3 texts as examples
            print(f" - {text[:100]}...")
        print(f"Total texts in cluster: {len(texts_in_cluster)}\n")

    # Visualize clusters
    visualize_graph_clusters(graph, cluster_labels)

    return cluster_labels, cluster_to_texts

def visualize_graph_clusters(graph, cluster_labels):
    """
    Visualize the graph clusters using NetworkX.
    """
    pos = nx.spring_layout(graph, seed=42)  # Layout for the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        graph,
        pos,
        node_color=cluster_labels,
        cmap=plt.cm.tab10,
        node_size=50,
        alpha=0.8,
        with_labels=False
    )
    plt.title("Graph Clustering Visualization")
    plt.show()

if __name__ == '__main__':
    author_dir = 'preprocessed_data_new'
    data = {}
    for author_file in os.listdir(author_dir):
        with open(os.path.join(author_dir, author_file), "r", encoding="utf-8") as file:
            data[author_file] = json.load(file)

    # Perform clustering
    cluster_labels, cluster_to_texts = graph_based_clustering(data)


    # Calculate coherence scores
    coherence_scores = calculate_cluster_coherence(cluster_to_texts)

    print(cluster_labels)
    # Print coherence scores
    for cluster, score in coherence_scores.items():
        print(f"Cluster {cluster} Coherence Score: {score}")
