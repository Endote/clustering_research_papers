import pandas as pd
from unidecode import unidecode

# --- Normalization helper ---
def normalize_author_name(name: str) -> str:
    """Ensure consistent name format for merging."""
    parts = name.strip().split()
    if len(parts) < 2:
        return unidecode(name)
    if parts[0].isupper() or parts[1][0].islower():
        parts = parts[::-1]
    first = unidecode(parts[0].capitalize())
    last = unidecode(" ".join(parts[1:]).capitalize())
    return f"{first} {last}"

# --- Load Data ---
semantic_df = pd.read_csv("semantic_nodes.csv")
community_df = pd.read_csv("detected_communities.csv")

# --- Normalize Names ---
semantic_df["NormId"] = semantic_df["Id"].apply(normalize_author_name)
community_df["NormId"] = community_df["Author"].apply(normalize_author_name)

# --- Merge ---
merged = pd.merge(semantic_df, community_df, on="NormId", how="inner")

# --- Optional: Save Merged File ---
merged.to_csv("merged_communities_semantics.csv", index=False, encoding="utf-8-sig")
print("✅ Merged file saved as 'merged_communities_semantics.csv'")


# Load the merged dataset
df = pd.read_csv("merged_communities_semantics.csv")

# Basic cleanup
df["SemanticCluster"] = df["SemanticCluster"].fillna("Noise")
df["Community"] = df["Community"].astype(str)

# 1. Count of Semantic Clusters per Community (diversity measure)
semantic_counts = df.groupby(["Community", "SemanticCluster"]).size().reset_index(name="Count")

# 2. Compute entropy (topic diversity) for each community
from scipy.stats import entropy

def compute_entropy(subdf):
    proportions = subdf["Count"] / subdf["Count"].sum()
    return entropy(proportions)

entropy_df = semantic_counts.groupby("Community").apply(compute_entropy).reset_index(name="TopicEntropy")

# 3. Most frequent semantic cluster in each community
dominant_topic = semantic_counts.sort_values("Count", ascending=False).drop_duplicates("Community")

# Merge dominant topic and entropy for summary
community_summary = pd.merge(entropy_df, dominant_topic, on="Community")
community_summary = community_summary.rename(columns={"SemanticCluster": "DominantSemanticCluster"})

# 4. Number of authors per community
author_counts = df["Community"].value_counts().reset_index()
author_counts.columns = ["Community", "NumAuthors"]

# Final summary
community_summary = pd.merge(community_summary, author_counts, on="Community")

print(community_summary)

