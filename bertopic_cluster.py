import json
import os
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from collections import defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel

def calculate_coherence_per_topic(topic_model, all_docs, assigned_topics, coherence='c_v'):
    """
    Calculate per-topic coherence for a BERTopic model.

    Parameters
    ----------
    topic_model : BERTopic
        A fitted BERTopic model.
    all_docs : list of str
        The list of all documents (strings) in the order used by BERTopic.
    assigned_topics : list of int
        Topic labels returned by BERTopic for each document. 
        (Same length/order as `all_docs`.)
    coherence : str
        The Gensim coherence measure, e.g. 'c_v', 'u_mass', 'c_npmi', etc.

    Returns
    -------
    dict
        A dictionary { topic_id: coherence_value }.
        If a topic has <8 docs, we store NaN.
    """
    # 1) Extract top words from each topic
    all_topics = topic_model.get_topics()  # e.g. {0: [(word, score), ...], 1: [...], ...}

    # 2) Tokenize all docs once (globally)
    tokenized_docs = [doc.split() for doc in all_docs]

    # 3) Build a dictionary from *all* tokens (global) for convenience
    dictionary = Dictionary(tokenized_docs)

    # 4) Prepare a results dict
    coherence_scores = {}

    # 5) Iterate over each topic in the BERTopic model
    for topic_id, word_score_pairs in all_topics.items():
        # if topic_id == -1:
        #     # Skip outlier/noise topic
        #     continue
        
        # (a) Gather top words for this topic (list of strings)
        topic_words = [w for w, _ in word_score_pairs]
        
        # (b) Collect the docs assigned to this topic
        docs_for_topic = []
        for i, t_id in enumerate(assigned_topics):
            if t_id == topic_id:
                docs_for_topic.append(tokenized_docs[i])

        # If too few docs, skip coherence
        if len(docs_for_topic) < 8:
            coherence_scores[topic_id] = float('nan')
            continue

        # (c) Build the corpus for these docs
        corpus_for_topic = [dictionary.doc2bow(doc) for doc in docs_for_topic]

        # (d) Build the CoherenceModel
        # We pass only a single "topic" in the `topics` parameter: [topic_words]
        coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=docs_for_topic,
            corpus=corpus_for_topic,
            dictionary=dictionary,
            coherence=coherence
        )
        
        # (e) Compute coherence for this topic
        coherence_scores[topic_id] = coherence_model.get_coherence()

    return coherence_scores

def calculate_topic_coherence(topic_model, combined_texts):
    # Extract topics from BERTopic
    topics = topic_model.get_topics()
    topic_keywords = [[word for word, _ in topics[topic]] for topic in topics if topic != -1]

    # Tokenize texts for coherence calculation
    tokenized_texts = [text.split() for text in combined_texts]

    # Create a Gensim dictionary and corpus
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # Calculate coherence using 'c_v' metric
    coherence_model = CoherenceModel(
        topics=topic_keywords,
        texts=tokenized_texts,
        corpus=corpus,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score

def cluster(papers):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract titles and abstracts and authors
    titles = [
        paper['title']
        for author_papers in papers.values()
        for paper in author_papers if paper.get('title') and paper.get('abstract') 
    ]
    abstracts = [
        paper['abstract']
        for author_papers in papers.values()
        for paper in author_papers if paper.get('title') and paper.get('abstract')
    ]
    authors = [
        paper['authors']
        for author_papers in papers.values()
        for paper in author_papers if paper.get('title') and paper.get('abstract')
    ]

    # Combine titles and abstracts for BERTopic clustering
    combined_texts = [
        f"{title}. {abstract}" for title, abstract in zip(titles, abstracts)
    ]
    combined_embeddings = model.encode(combined_texts, show_progress_bar=True)

    # Custom HDBSCAN model
    hdbscan_model = HDBSCAN(min_cluster_size=12, min_samples=7, metric='euclidean', cluster_selection_method='eom')

    # Apply BERTopic
    topic_model = BERTopic(hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(combined_texts, combined_embeddings)
    topic_model.update_topics(combined_texts, top_n_words=7)

    # Print topics
    print(topic_model.get_topic_info())

    # Now compute coherence for each topic:
    coherence_dict = calculate_coherence_per_topic(
        topic_model=topic_model,
        all_docs=combined_texts,
        assigned_topics=topics,
        coherence='c_v'   # or 'u_mass', 'c_npmi', etc.
    )

    # Print results
    for t_id, score in coherence_dict.items():
        print(f"Topic {t_id} => coherence: {score:.4f}")

    ## Get coherence for all topics together
    coherence_score = calculate_topic_coherence(topic_model, combined_texts)
    print(f"Topic Coherence Score: {coherence_score}")

    # Visualize clustering
    visualize_embeddings(combined_embeddings, topics)

    # Create a mapping of topics to titles and authors
    topic_to_titles_and_authors = {}
    for idx, topic in enumerate(topics):
        if topic not in topic_to_titles_and_authors:
            topic_to_titles_and_authors[topic] = []
        topic_to_titles_and_authors[topic].append({"title": titles[idx], "authors": authors[idx]})

    # Return authors as well, so we can do an "author -> (topic counts)" mapping outside
    return topics, probs, topic_to_titles_and_authors, authors

def visualize_embeddings(combined_embeddings, topics):
    """
    Visualize BERTopic clustering results on a 2D plot, with topics separated on the y-axis.
    """
    # Reduce dimensionality to 1D (use only the first dimension for simplicity)
    x = combined_embeddings[:, 0]  # First dimension of the combined embeddings

    # Use topic IDs as the y-axis
    y = topics  # Assign topic IDs directly as y values

    # Plot
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(x, y, c=topics, cmap='tab10', s=50, alpha=0.7)
    plt.colorbar(scatter, label="Topic ID")
    plt.title("BERTopic Clustering Visualization (Topics Separated on Y-Axis)")
    plt.xlabel("Reduced Combined Embedding (1st Dimension)")
    plt.ylabel("Topic ID")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def load_data_deduplicated(author_dir):
    """
    Load all author files from `author_dir` and deduplicate papers across authors.
    Returns a dict: { author_file: [paper1, paper2, ...], ... }
    """
    data = {}
    seen = set()  # will store a key like (title, abstract) or (doi) to track duplicates

    for author_file in os.listdir(author_dir):
        with open(os.path.join(author_dir, author_file), "r", encoding="utf-8") as file:
            papers = json.load(file)  # list of dicts, each dict is a paper

        deduped_papers = []
        for paper in papers:
            #  key: (title.lower(), abstract.lower())
            if paper['title'] and paper['abstract']:
                title = paper.get('title', '').strip().lower()
                abstract = paper.get('abstract', '').strip().lower()
                dedup_key = (title, abstract)

                if dedup_key not in seen:
                    seen.add(dedup_key)
                    deduped_papers.append(paper)

        # Store the deduplicated list for this author file
        data[author_file] = deduped_papers

    return data

if __name__ == '__main__':
    author_dir = 'preprocessed_data_new'
    
    # Load deduplicated data
    data = load_data_deduplicated(author_dir)

    # Now perform clustering on this deduplicated data
    topics, probs, topic_to_titles_and_authors, authors = cluster(data)

    # author_topic_counts[author_name][topic_id] = number_of_papers
    author_topic_counts = defaultdict(lambda: defaultdict(int))

    # 'topics' and 'authors' have the same length; each index 'i' is one paper
    # authors[i] is typically a list of authors for that paper
    for i, topic_id in enumerate(topics):
        # authors[i] might be a list of strings
        for auth in authors[i]:
            author_topic_counts[auth][topic_id] += 1

    # Print the results: only show topic counts > 0
    print("\n=== Author -> Topic Counts ===")
    for author, topic_dict in author_topic_counts.items():
        # You might skip authors with no actual papers, but here we only show if there's something > 0
        print(f"\nAuthor: {author}")
        for t_id, count in topic_dict.items():
            if count > 0:
                print(f"  Topic {t_id}: {count} paper(s)")
