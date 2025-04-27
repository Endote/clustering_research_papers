import pandas as pd

# --- Helper Functions ---
def normalize_string(text):
    """Normalize string for consistent matching: lowercase, remove punctuation."""
    return (
        str(text)
        .lower()
        .replace("-", " ")
        .replace(".", "")
        .replace(",", "")
        .strip()
    )

def build_subset_mapping(id_list):
    """Build a mapping of subset names to their canonical full names."""
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
    subset_matches_df = pd.DataFrame(list(set(subset_matches)), columns=["Longer Name", "Subset Name"])
    return dict(subset_matches_df[["Subset Name", "Longer Name"]].values)

def build_abbreviation_mapping(id_list):
    """Handle abbreviated names like 'Lastname A' -> 'Firstname Lastname'."""
    abbrev_map = {}
    for name in id_list:
        tokens = name.split()
        if len(tokens) == 2 and len(tokens[1]) == 1:
            last, initial = tokens
            for full in id_list:
                full_tokens = full.split()
                if len(full_tokens) >= 2 and last == full_tokens[-1] and full_tokens[0].startswith(initial):
                    abbrev_map[name] = full
    return abbrev_map

# --- Load Data ---
semantic_nodes = pd.read_csv("semantic_nodes_normalized.csv")
edge_list = pd.read_csv("gephi_edge_list_normalized.csv")

# --- Normalize Semantic Nodes ---
semantic_nodes["Id_clean"] = semantic_nodes["Id"].apply(normalize_string)
semantic_ids = semantic_nodes["Id_clean"].unique().tolist()

# Subset and abbreviation mapping
subset_map = build_subset_mapping(semantic_ids)
abbrev_map = build_abbreviation_mapping(semantic_ids)

# Apply both mappings, then title case
semantic_nodes["CanonicalId"] = (
    semantic_nodes["Id_clean"]
    .replace(subset_map)
    .replace(abbrev_map)
    .apply(lambda x: x.title())
)

# Deduplicate semantic_nodes
semantic_deduped = (
    semantic_nodes.sort_values("TotalPapers", ascending=False)
    .drop_duplicates(subset=["CanonicalId"])
    .drop(columns=["Id_clean"])
    .rename(columns={"CanonicalId": "Id"})
)

# --- Normalize Edge List ---
edge_list["Source_clean"] = edge_list["Source"].apply(normalize_string)
edge_list["Target_clean"] = edge_list["Target"].apply(normalize_string)

edge_list["Source_final"] = (
    edge_list["Source_clean"]
    .replace(subset_map)
    .replace(abbrev_map)
    .apply(lambda x: x.title())
)
edge_list["Target_final"] = (
    edge_list["Target_clean"]
    .replace(subset_map)
    .replace(abbrev_map)
    .apply(lambda x: x.title())
)

# Drop unused columns and rename for Gephi
edge_list_final = edge_list[["Source_final", "Target_final", "Weight"]].rename(
    columns={"Source_final": "Source", "Target_final": "Target"}
)

# --- Save Results ---
semantic_deduped.to_csv("final_semantic_nodes.csv", index=False, encoding="utf-8-sig")
edge_list_final.to_csv("final_gephi_edge_list.csv", index=False, encoding="utf-8-sig")

print("✅ Normalization complete. Final files saved.")
