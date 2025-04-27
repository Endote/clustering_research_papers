import os
import json
import pandas as pd
from collections import defaultdict
from itertools import combinations
import unicodedata

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

data_dir = "preprocessed_data_new"
coauthor_counts = defaultdict(int)
authors_all = set()

for file in os.listdir(data_dir):
    if file.endswith(".json"):
        with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
            papers = json.load(f)
            for paper in papers:
                if "authors" in paper and paper["authors"]:
                    authors = [normalize_author_name_verbose(a.strip()) for a in paper["authors"].split(",") if a.strip()]
                    authors_all.update(authors)
                    for a1, a2 in combinations(sorted(authors), 2):
                        coauthor_counts[(a1, a2)] += 1

sorted_authors = sorted(authors_all)
adj_df = pd.DataFrame(0, index=sorted_authors, columns=sorted_authors)

for (a1, a2), count in coauthor_counts.items():
    adj_df.loc[a1, a2] = count
    adj_df.loc[a2, a1] = count

adj_df.to_csv("coauthorship_matrix_raw.csv", encoding="utf-8-sig")
print("✅ Saved coauthorship matrix.")

edges = []
for i, a1 in enumerate(adj_df.index):
    for j, a2 in enumerate(adj_df.columns):
        if j <= i: continue
        weight = adj_df.iloc[i, j]
        if weight > 0:
            edges.append((a1, a2, weight))

pd.DataFrame(edges, columns=["Source", "Target", "Weight"]).to_csv("gephi_edge_list.csv", index=False, encoding="utf-8-sig")
print("✅ Saved Gephi edge list.")
