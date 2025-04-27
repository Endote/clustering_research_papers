import pandas as pd

# Load the semantic_nodes dataset
semantic_nodes = pd.read_csv("semantic_nodes.csv")

# Normalize for consistent comparison
semantic_nodes["Id_clean"] = (
    semantic_nodes["Id"]
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.strip()
)

# Build list of unique cleaned names
id_list = semantic_nodes["Id_clean"].unique().tolist()

# Step 1: Detect subset name cases
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

# Step 2: Merge based on subset mapping
subset_to_canonical = dict(subset_matches_df[["Potential Subset Name", "Longer Name"]].values)
semantic_nodes["CanonicalId"] = semantic_nodes["Id_clean"].replace(subset_to_canonical)

# Step 3: Handle abbreviated names (e.g., "Lastname A" vs "Firstname Lastname")
abbrev_map = {}

for name in id_list:
    tokens = name.split()
    if len(tokens) == 2 and len(tokens[1]) == 1:  # format "lastname a"
        abbrev_last, abbrev_initial = tokens
        for full in id_list:
            full_tokens = full.split()
            if len(full_tokens) >= 2 and abbrev_last == full_tokens[-1] and full_tokens[0].startswith(abbrev_initial):
                abbrev_map[name] = full

# Apply abbreviated mapping only to unmatched canonical IDs
semantic_nodes["CanonicalId"] = semantic_nodes["CanonicalId"].replace(abbrev_map)

# Deduplicate on updated CanonicalId
deduplicated = (
    semantic_nodes.sort_values("TotalPapers", ascending=False)
    .drop_duplicates(subset=["CanonicalId"])
    .drop(columns=["Id_clean"])
    .rename(columns={"CanonicalId": "Id"})
)

# Save the final cleaned and deduplicated semantic_nodes file
output_path = "semantic_nodes_normalized.csv"
deduplicated.to_csv(output_path, index=False, encoding="utf-8-sig")
