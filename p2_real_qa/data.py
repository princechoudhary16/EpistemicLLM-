import ast
import random
import re
import pandas as pd
from pathlib import Path
import config

def load_and_snapshot_dataset(snapshot_name: str) -> pd.DataFrame:
    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SIMPLEQA_URL)
    df.to_csv(f"{config.RAW_DIR}/{snapshot_name}.csv", index=False)
    df["metadata"] = df["metadata"].apply(ast.literal_eval)
    df["question_id"] = df.index
    return df

def select_pilot_questions(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    unique = df.copy()
    unique["_split_group"] = _group_keys(unique)
    unique = unique.drop_duplicates("_split_group")
    if n > len(unique):
        raise ValueError(f"requested {n} pilot questions, but only {len(unique)} unique groups exist")
    return unique.sample(n=n, random_state=config.PILOT_SEED).sort_values("question_id").drop(
        columns="_split_group").reset_index(drop=True)


def _normalized_question(question):
    text = str(question or "").casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _group_keys(frame):
    """Connect exact duplicates and reviewed family IDs into stable group keys."""
    parent = list(range(len(frame)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    alias_owner, aliases_by_row = {}, []
    for position, (_, row) in enumerate(frame.iterrows()):
        aliases = {f"question:{_normalized_question(row.get('problem', ''))}"}
        group_id = row.get("group_id")
        if group_id is not None and not pd.isna(group_id) and str(group_id).strip():
            aliases.add(f"family:{group_id}")
        aliases_by_row.append(aliases)
        for alias in aliases:
            if alias in alias_owner:
                union(position, alias_owner[alias])
            else:
                alias_owner[alias] = position

    component_aliases = {}
    for position, aliases in enumerate(aliases_by_row):
        component_aliases.setdefault(find(position), set()).update(aliases)
    stable_key = {root: min(aliases) for root, aliases in component_aliases.items()}
    return pd.Series([stable_key[find(i)] for i in range(len(frame))], index=frame.index)

def select_main_splits(df: pd.DataFrame, pilot: pd.DataFrame):
    """Select about 1,000 questions without splitting duplicate/family groups.

    Exact normalized question duplicates are grouped automatically. If P2 has manually
    reviewed related-question families, a `group_id` column keeps those together too.
    Group sizes can make the returned split counts differ slightly from 500/200/300.
    """
    all_rows = df.copy()
    all_rows["_split_group"] = _group_keys(all_rows)
    pilot_ids = set(pilot["question_id"])
    pilot_groups = set(all_rows.loc[all_rows["question_id"].isin(pilot_ids), "_split_group"])
    remaining = all_rows.loc[~all_rows["question_id"].isin(pilot_ids)].copy()
    remaining = remaining.loc[~remaining["_split_group"].isin(pilot_groups)]

    groups = {}
    for group_key, rows in remaining.groupby("_split_group", sort=False):
        groups[group_key] = rows
    if len(remaining) < 1000:
        raise ValueError(f"need 1,000 remaining questions; found {len(remaining)}")
    if not groups:
        raise ValueError("no unassigned question groups remain")

    targets = {"train_probe": 500, "validation": 200, "locked_test": 300}
    group_keys = list(groups)
    random.Random(config.MAIN_SPLIT_SEED).shuffle(group_keys)
    selected, selected_count = [], 0
    for key in group_keys:
        selected.append(key)
        selected_count += len(groups[key])
        if selected_count >= 1000:
            break
    if len(selected) < len(targets):
        raise ValueError("need at least three disjoint question groups for train/validation/test splits")

    # Allocate whole groups toward the planned 500/200/300 split sizes.
    split_groups = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    allocation_order = sorted(selected, key=lambda key: (-len(groups[key]), selected.index(key)))
    for key in allocation_order:
        split_name = max(targets, key=lambda name: (targets[name] - counts[name]) / targets[name])
        split_groups[split_name].append(key)
        counts[split_name] += len(groups[key])

    return {
        name: pd.concat([groups[key] for key in keys]).drop(columns="_split_group")
        .sort_values("question_id").reset_index(drop=True)
        for name, keys in split_groups.items()
    }

def save_manifest(split_df: pd.DataFrame, name: str):
    split_df[["question_id", "problem", "answer"]].to_csv(f"{config.MANIFEST_DIR}/p2_{name}.csv", index=False)
