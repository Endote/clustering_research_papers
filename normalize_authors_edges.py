import pandas as pd

# Load edge list and semantic_nodes
edge_df = pd.read_csv("gephi_edge_list.csv")
semantic_df = pd.read_csv("semantic_nodes.csv")

# Normalize semantic node IDs
semantic_df["Id_clean"] = (
    semantic_df["Id"]
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.strip()
)

# Unique ID list
id_list = semantic_df["Id_clean"].unique().tolist()

# Step 1: Detect subset name matches
subset_matches = []
for name in id_list:
    name_tokens = name.split()
    if len(name_tokens) < 2:
        continue
    for other in id_list:
        if name == other:
            continue
        other_tokens = other.split()
        if all(tok in name_tokens for tok in other_tokens):
            subset_matches.append((name, other))

subset_matches = list(set(subset_matches))
subset_matches_df = pd.DataFrame(subset_matches, columns=["Longer Name", "Potential Subset Name"])

# Step 2: Create canonical map from subset matches
subset_to_canonical = dict(subset_matches_df[["Potential Subset Name", "Longer Name"]].values)

# Step 3: Add abbreviation-based mapping
abbrev_map = {}
for name in id_list:
    tokens = name.split()
    if len(tokens) == 2 and len(tokens[1]) == 1:
        abbrev_last, abbrev_initial = tokens
        for full in id_list:
            full_tokens = full.split()
            if len(full_tokens) >= 2 and abbrev_last == full_tokens[-1] and full_tokens[0].startswith(abbrev_initial):
                abbrev_map[name] = full

# Merge subset and abbreviation maps
canonical_map = {**subset_to_canonical, **abbrev_map}

# Apply normalization to edge list
def normalize_name(name):
    norm = (
        str(name).lower()
        .replace(".", "")
        .replace(",", "")
        .strip()
    )
    return canonical_map.get(norm, norm)

edge_df["Source_normalized"] = edge_df["Source"].apply(normalize_name)
edge_df["Target_normalized"] = edge_df["Target"].apply(normalize_name)

# Save normalized edge list
normalized_edge_path = "gephi_edge_list_normalized.csv"
edge_df.to_csv(normalized_edge_path, index=False, encoding="utf-8-sig")

