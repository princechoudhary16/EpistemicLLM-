"""P2 real-QA dataset preparation (SimpleQA).

    python -m p2_real_qa.data --snapshot data/raw/simpleqa_snapshot_2026-09-28.csv --version v0
    python -m p2_real_qa.data --download data/raw/simpleqa_snapshot_2026-10-16.csv     # fetch a NEW snapshot, never overwrites

Outputs (deterministic for a given snapshot + seeds):
    data/real_qa/simpleqa_<version>/{pilot,train_probe,validation,locked_test}.jsonl
        one row per question in the format common.evaluate.run_eval expects
        (question_id, source_dataset, dataset_version, split, question, reference_answer)
        plus P2/P3-side metadata (topic, answer_type, group_id).
    data/manifests/real_qa_simpleqa_<version>/manifest.jsonl   ids + split + group only
    data/manifests/real_qa_simpleqa_<version>/audit.json       counts, seeds, hashes, validation results
"""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2_real_qa.validate import run_checks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "simpleqa"
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
PILOT_SEED = 20260928          # unchanged from the first pilot, so the SAME 200 questions are selected
MAIN_SPLIT_SEED = 20261004
PILOT_N = 200
SPLIT_SIZES = (("train_probe", 500), ("validation", 200), ("locked_test", 300))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- stage 1: snapshot
def download_snapshot(url: str, path) -> str:
    """Downloads the official CSV once. Refuses to overwrite: a snapshot is evidence."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; pick a new snapshot name instead of overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(url).to_csv(path, index=False)
    return sha256(path)


_WIKI = re.compile(r"https?://[a-z.]*wikipedia\.org/wiki/[^\s#]+", re.I)


def _group_key(urls):
    """First Wikipedia article URL. Some SimpleQA entries pack several URLs into ONE string separated by newlines,
    so we regex the URL out instead of using the whole string."""
    for u in urls:
        m = _WIKI.search(u)
        if m:
            return m.group(0).lower()
    return None


def normalized_question(q: str) -> str:
    """Lower-case, punctuation removed, whitespace collapsed: 'Who is Ann?' == 'Who is Ann!'."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(q).lower())).strip()


def merge_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Questions with the same normalized text are forced into ONE group (union-find over group_ids), so a duplicate
    can never land in a different split from its twin. Deterministic: the smallest group_id string is the root."""
    df = df.copy()
    parent = {g: g for g in df.group_id.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in df.groupby(df.problem.map(normalized_question)):
        gs = list(grp.group_id.unique())
        for g in gs[1:]:
            ra, rb = find(gs[0]), find(g)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    df["group_id"] = df.group_id.map(find)
    return df


def apply_manual_groups(df: pd.DataFrame, path) -> pd.DataFrame:
    """Optional human review: a CSV with columns question_id,group_id forces those questions into the given family
    (e.g. several rewordings of one fact). Applied BEFORE duplicate merging."""
    if path is None or not Path(path).exists():
        return df
    m = pd.read_csv(path).set_index("question_id").group_id.to_dict()
    df = df.copy()
    df["group_id"] = [m.get(q, g) for q, g in zip(df.question_id, df.group_id)]
    return df


def load_snapshot(path, manual_groups=None) -> pd.DataFrame:
    """Reads the CSV. question_id = row position in the official file (stable as long as the file is unchanged)."""
    df = pd.read_csv(path)
    df["question_id"] = df.index
    md = df["metadata"].apply(ast.literal_eval)
    df["topic"] = md.map(lambda m: m.get("topic"))
    df["answer_type"] = md.map(lambda m: m.get("answer_type"))
    # group_id = first Wikipedia page cited; questions about the same page must never straddle splits (P3 leakage).
    df["group_id"] = md.map(lambda m: _group_key(m.get("urls", [])))
    missing = df["group_id"].isna()
    df.loc[missing, "group_id"] = "q" + df.loc[missing, "question_id"].astype(str)
    df["answer"] = df["answer"].astype(str).str.strip()
    df = merge_duplicate_groups(apply_manual_groups(df, manual_groups))
    return df.drop(columns=["metadata"])


# ---------------------------------------------------------------- stage 2: selection
def select_pilot(df: pd.DataFrame, n: int = PILOT_N, seed: int = PILOT_SEED) -> pd.DataFrame:
    """Identical to the original pilot selection (same seed, same call) so earlier audit work stays valid."""
    unique = df[~df.problem.map(normalized_question).duplicated()]       # keep first of any duplicated wording
    return unique.sample(n=n, random_state=seed).sort_values("question_id").reset_index(drop=True)


def group_split(df, exclude_ids, seed=MAIN_SPLIT_SEED, sizes=SPLIT_SIZES) -> dict:
    """Group-aware, pilot-safe split.
    1. Drop the pilot questions AND every question sharing a group with one (a pilot twin must not leak).
    2. Shuffle the remaining GROUPS with a seed.
    3. Fill each split exactly to its size, taking a group only if it still fits. A group is never cut in half and
       never straddles two splits; a group too big for one split is simply offered to the next."""
    pilot_groups = set(df[df.question_id.isin(set(exclude_ids))].group_id)
    pool = df[~df.group_id.isin(pilot_groups)]
    remaining = pool.group_id.drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    by_group = {g: d for g, d in pool.groupby("group_id")}
    out = {}
    for name, n in sizes:
        taken, total, leftover = [], 0, []
        for g in remaining:
            if total < n and total + len(by_group[g]) <= n:
                taken.append(by_group[g])
                total += len(by_group[g])
            else:
                leftover.append(g)
        if total != n:
            raise RuntimeError(f"cannot fill split '{name}' to exactly {n} questions (got {total}); pool too small")
        remaining = leftover
        out[name] = pd.concat(taken).sort_values("question_id").reset_index(drop=True)
    return out


# ---------------------------------------------------------------- stage 3: records
def qid(i: int) -> str:
    return f"sqa-{int(i):05d}"       # string IDs cannot collide with other datasets when files are merged in Week 5


def to_records(df: pd.DataFrame, split: str, dataset_version: str) -> list:
    return [dict(question_id=qid(r.question_id), source_dataset=SOURCE, dataset_version=dataset_version,
                 split=split, question=r.problem, reference_answer=r.answer,
                 topic=r.topic, answer_type=r.answer_type, group_id=r.group_id)
            for r in df.itertuples()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- stage 4: build everything
def build(snapshot, version="v0", data_root=None, manifest_root=None, pilot_n=PILOT_N, sizes=SPLIT_SIZES):
    data_dir = Path(data_root or REPO_ROOT / "data" / "real_qa") / f"{SOURCE}_{version}"
    man_dir = Path(manifest_root or REPO_ROOT / "data" / "manifests") / f"real_qa_{SOURCE}_{version}"
    ds_version = f"{SOURCE}-{version}"

    df = load_snapshot(snapshot)
    pilot = select_pilot(df, pilot_n)
    splits = {"pilot": pilot, **group_split(df, pilot.question_id, MAIN_SPLIT_SEED, sizes)}
    records = {name: to_records(d, name, ds_version) for name, d in splits.items()}

    for name, rows in records.items():
        write_jsonl(data_dir / f"{name}.jsonl", rows)
    write_jsonl(man_dir / "manifest.jsonl",
                [{k: r[k] for k in ("question_id", "split", "group_id")} for rows in records.values() for r in rows])

    checks = run_checks(records, {"pilot": pilot_n, **dict(sizes)})
    audit = {
        "dataset_version": ds_version, "snapshot_file": Path(snapshot).name, "snapshot_sha256": sha256(snapshot),
        "n_snapshot_rows": len(df), "pilot_seed": PILOT_SEED, "main_split_seed": MAIN_SPLIT_SEED,
        "questions_per_split": {k: len(v) for k, v in records.items()},
        "groups_per_split": {k: len({r["group_id"] for r in v}) for k, v in records.items()},
        "answer_type_per_split": {k: pd.Series([r["answer_type"] for r in v]).value_counts().to_dict()
                                  for k, v in records.items()},
        "checks": checks,
        "file_sha256": {p.name: sha256(p) for p in sorted(data_dir.glob("*.jsonl"))},
    }
    (man_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshot", default="data/raw/simpleqa_snapshot_2026-09-28.csv")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--download", metavar="NEW_SNAPSHOT_PATH", help="download SimpleQA to this (new) path and exit")
    a = ap.parse_args(argv)
    if a.download:
        print("saved", a.download, "sha256", download_snapshot(SIMPLEQA_URL, REPO_ROOT / a.download))
        return
    audit = build(REPO_ROOT / a.snapshot, a.version)
    for c in audit["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print("questions per split:", audit["questions_per_split"])
    sys.exit(0 if all(c["ok"] for c in audit["checks"]) else 1)


if __name__ == "__main__":
    main()
