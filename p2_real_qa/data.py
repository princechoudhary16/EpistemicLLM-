"""P2 real-QA dataset preparation (SimpleQA and TriviaQA).

    python -m p2_real_qa.data --snapshot data/raw/simpleqa_snapshot_2026-09-28.csv --version v0
    python -m p2_real_qa.data --download data/raw/simpleqa_snapshot_2026-10-16.csv     # fetch a NEW snapshot, never overwrites

    python -m p2_real_qa.data --download-triviaqa data/raw/triviaqa_rc_nocontext_validation_0f7faf33.jsonl   # once
    python -m p2_real_qa.data --source triviaqa --version v0                                                  # build splits

TriviaQA outputs go to data/real_qa/triviaqa_<version>/ and data/manifests/real_qa_triviaqa_<version>/, same layout.

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


def make_qid(source: str, raw_id) -> str:
    """SimpleQA: sqa-00004. TriviaQA: triviaqa-<official id>, the same IDs P4 uses, so files join across tracks."""
    return qid(raw_id) if source == SOURCE else f"{source}-{raw_id}"


def to_records(df: pd.DataFrame, split: str, dataset_version: str, source: str = SOURCE) -> list:
    rows = []
    for r in df.itertuples():
        row = dict(question_id=make_qid(source, r.question_id), source_dataset=source, dataset_version=dataset_version,
                   split=split, question=r.problem, reference_answer=r.answer,
                   topic=r.topic, answer_type=r.answer_type, group_id=r.group_id)
        if "aliases" in df.columns:          # TriviaQA: the same answers as a list, for code that reads `aliases` (P4)
            row["aliases"] = list(r.aliases)
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- stage 4: build everything
def build(snapshot, version="v0", data_root=None, manifest_root=None, pilot_n=PILOT_N, sizes=SPLIT_SIZES):
    df = load_snapshot(snapshot)
    pilot = select_pilot(df, pilot_n)
    splits = {"pilot": pilot, **group_split(df, pilot.question_id, MAIN_SPLIT_SEED, sizes)}
    return write_release(SOURCE, version, df, splits, snapshot, PILOT_SEED, MAIN_SPLIT_SEED,
                         data_root, manifest_root, pilot_n, sizes)


def write_release(source, version, df, splits, snapshot, pilot_seed, main_split_seed,
                  data_root=None, manifest_root=None, pilot_n=PILOT_N, sizes=SPLIT_SIZES, extra_audit=None):
    """Writes the split files, manifest and audit.json for one dataset release; returns the audit dict."""
    data_dir = Path(data_root or REPO_ROOT / "data" / "real_qa") / f"{source}_{version}"
    man_dir = Path(manifest_root or REPO_ROOT / "data" / "manifests") / f"real_qa_{source}_{version}"
    ds_version = f"{source}-{version}"
    records = {name: to_records(d, name, ds_version, source) for name, d in splits.items()}

    for name, rows in records.items():
        write_jsonl(data_dir / f"{name}.jsonl", rows)
    write_jsonl(man_dir / "manifest.jsonl",
                [{k: r[k] for k in ("question_id", "split", "group_id")} for rows in records.values() for r in rows])

    checks = run_checks(records, {"pilot": pilot_n, **dict(sizes)})
    audit = {
        "dataset_version": ds_version, "snapshot_file": Path(snapshot).name, "snapshot_sha256": sha256(snapshot),
        "n_snapshot_rows": len(df), "pilot_seed": pilot_seed, "main_split_seed": main_split_seed,
        "questions_per_split": {k: len(v) for k, v in records.items()},
        "groups_per_split": {k: len({r["group_id"] for r in v}) for k, v in records.items()},
        "answer_type_per_split": {k: pd.Series([r["answer_type"] for r in v]).value_counts().to_dict()
                                  for k, v in records.items()},
        "checks": checks,
        "file_sha256": {p.name: sha256(p) for p in sorted(data_dir.glob("*.jsonl"))},
        **(extra_audit or {}),
    }
    (man_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


# ---------------------------------------------------------------- TriviaQA (second real-QA source)
TRIVIAQA = "triviaqa"
TRIVIAQA_REPO = "mandarjoshi/trivia_qa"
TRIVIAQA_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"    # same pin as confidence_calibration/data.py (P4)
TRIVIAQA_FILE = "rc.nocontext/validation-00000-of-00001.parquet"
TRIVIAQA_SNAPSHOT = "data/raw/triviaqa_rc_nocontext_validation_0f7faf33.jsonl"
P4_SAMPLE = "data/triviaqa/validation_sample_n300_seed0.jsonl"     # P4's calibration questions: kept out of every split
TQA_PILOT_SEED = 20261029
TQA_MAIN_SPLIT_SEED = 20261030
MIN_ALIAS_CHARS = 2      # a non-canonical alias shorter than this (after normalising) would match almost any answer


def download_triviaqa_snapshot(path) -> str:
    """Downloads the pinned TriviaQA rc.nocontext validation file once and saves a compact JSONL snapshot
    (one row per question: question_id, question, value, aliases, type). Refuses to overwrite, like SimpleQA.
    Needs huggingface_hub and pyarrow (both in confidence_calibration/requirements.txt)."""
    from huggingface_hub import hf_hub_download

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; pick a new snapshot name instead of overwriting")
    parquet = hf_hub_download(TRIVIAQA_REPO, TRIVIAQA_FILE, repo_type="dataset", revision=TRIVIAQA_REVISION)
    df = pd.read_parquet(parquet).sort_values("question_id").reset_index(drop=True)
    rows = [dict(question_id=str(r.question_id), question=str(r.question), value=str(r.answer["value"]),
                 aliases=[str(a) for a in r.answer["aliases"]], type=str(r.answer.get("type") or ""))
            for r in df.itertuples()]
    write_jsonl(path, rows)
    return sha256(path)


def clean_aliases(value, aliases, question="") -> tuple[list, dict]:
    """Canonical value first, then the other aliases, de-duplicated by their normalised form. Each alias with a
    bracketed label also adds its bare form: Wikipedia-style aliases like "Roots (book)" or "Mercury (album)" would
    otherwise never match a plain answer ("Roots").

    Dropped (counted in the audit): aliases that normalise to fewer than MIN_ALIAS_CHARS characters (digits are
    kept: the grader has its own number rules), aliases already
    written in the question (restating the question must not earn credit), and "||" inside an alias (it is the
    grader's separator). The canonical value is always kept."""
    from common.grading import ALIAS_SEP, contains_whole, normalize

    nq = normalize(question)
    nq_bare = normalize(re.sub(r"(?<=\w)['\u2019]s\b", "", str(question)))    # "Elephant Man's" -> "Elephant Man"
    kept, seen = [], set()
    dropped = {"too_short": 0, "in_question": 0}
    candidates = []
    for a in [value, *aliases]:            # "Roots (book)" also yields "Roots"; the canonical value stays first
        candidates.append(a)
        bare = re.sub(r"\s*\([^()]*\)", "", str(a)).strip()
        if bare and bare != str(a).strip():
            candidates.append(bare)
    for i, a in enumerate(candidates):
        a = re.sub(r"\s+", " ", str(a).replace(ALIAS_SEP.strip(), " ")).strip()
        n = normalize(a)
        if not n or n in seen:
            continue
        if i > 0 and len(n) < MIN_ALIAS_CHARS and not n.isdigit():     # "8" for "Eight" is kept
            dropped["too_short"] += 1
            continue
        if i > 0 and (contains_whole(nq, n) or contains_whole(nq_bare, n)):
            dropped["in_question"] += 1
            continue
        seen.add(n)
        kept.append(a)
    return kept, dropped


def load_triviaqa_snapshot(path) -> tuple[pd.DataFrame, dict]:
    """Snapshot JSONL -> the same frame shape load_snapshot gives for SimpleQA.

    One row per question_id: the rc files repeat most questions (once per evidence source); the first copy is kept.
    reference_answer = every kept alias joined with " || " (common.grading accepts any of them).
    group_id = the normalised canonical answer: questions with the same answer ("France") never straddle splits,
    then exact-duplicate question texts are merged into one group as for SimpleQA."""
    from common.grading import ALIAS_SEP, normalize

    rows, dropped = [], {"too_short": 0, "in_question": 0}
    seen, n_rows, n_conflicting = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            # The rc files list most questions twice (Wikipedia and web evidence). Keep the FIRST row per ID
            # (the snapshot is sorted by ID, so this is deterministic); count copies whose text/answer differ.
            if r["question_id"] in seen:
                n_conflicting += seen[r["question_id"]] != (r["question"], r["value"], sorted(r.get("aliases", [])))
                continue
            seen[r["question_id"]] = (r["question"], r["value"], sorted(r.get("aliases", [])))
            aliases, d = clean_aliases(r["value"], r.get("aliases", []), r["question"])
            for k in dropped:
                dropped[k] += d[k]
            rows.append(dict(question_id=r["question_id"], problem=r["question"].strip(), answer=ALIAS_SEP.join(aliases),
                             aliases=aliases, topic=None, answer_type=r.get("type") or None,
                             group_id="ans:" + (normalize(r["value"]) or r["question_id"])))
    df = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    dedup = {"snapshot_rows": n_rows, "unique_question_ids": len(df), "repeated_rows_dropped": n_rows - len(df),
             "repeated_rows_with_different_content": int(n_conflicting)}
    return merge_duplicate_groups(df), {"aliases_dropped": dropped, "dedup": dedup}


def read_excluded_ids(path) -> set:
    """IDs from another track's question file (e.g. P4's sample), without the 'triviaqa-' prefix."""
    if not path or not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["question_id"].removeprefix(f"{TRIVIAQA}-") for l in f if l.strip()}


def build_triviaqa(snapshot, version="v0", exclude_file=P4_SAMPLE, data_root=None, manifest_root=None,
                   pilot_n=PILOT_N, sizes=SPLIT_SIZES):
    """Same release as build(), from the TriviaQA snapshot. Every question in exclude_file (P4's calibration sample)
    and every question sharing a group with one is removed BEFORE selection, so none can reach any split."""
    df, alias_audit = load_triviaqa_snapshot(snapshot)
    excluded = read_excluded_ids(REPO_ROOT / exclude_file if exclude_file and not Path(exclude_file).is_absolute()
                                 else exclude_file)
    excluded_groups = set(df[df.question_id.isin(excluded)].group_id)
    pool = df[~df.group_id.isin(excluded_groups)]
    pilot = select_pilot(pool, pilot_n, TQA_PILOT_SEED)
    splits = {"pilot": pilot, **group_split(pool, pilot.question_id, TQA_MAIN_SPLIT_SEED, sizes)}
    extra = {
        "source_repo": f"{TRIVIAQA_REPO}@{TRIVIAQA_REVISION}", "source_file": TRIVIAQA_FILE,
        "excluded_file": str(exclude_file) if exclude_file else None,
        "excluded_ids_requested": len(excluded),
        "excluded_ids_not_in_snapshot": len(excluded - set(df.question_id)),
        "excluded_questions_incl_group": int(len(df) - len(pool)),
        "aliases_per_question_mean": round(float(df.aliases.map(len).mean()), 2),
        **alias_audit,
    }
    audit = write_release(TRIVIAQA, version, df, splits, snapshot, TQA_PILOT_SEED, TQA_MAIN_SPLIT_SEED,
                          data_root, manifest_root, pilot_n, sizes, extra)
    leaked = excluded & {q.removeprefix(f"{TRIVIAQA}-") for d in splits.values() for q in map(str, d.question_id)}
    audit["checks"].append({"name": "excluded_ids_absent", "ok": not leaked, "detail": f"{len(leaked)} excluded IDs in splits"})
    man_dir = Path(manifest_root or REPO_ROOT / "data" / "manifests") / f"real_qa_{TRIVIAQA}_{version}"
    (man_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=(SOURCE, TRIVIAQA), default=SOURCE)
    ap.add_argument("--snapshot", help="default: data/raw/simpleqa_snapshot_2026-09-28.csv, or the TriviaQA snapshot")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--exclude", default=P4_SAMPLE, help="TriviaQA only: question file whose IDs stay out of every split")
    ap.add_argument("--download", metavar="NEW_SNAPSHOT_PATH", help="download SimpleQA to this (new) path and exit")
    ap.add_argument("--download-triviaqa", metavar="NEW_SNAPSHOT_PATH", help="download pinned TriviaQA to this (new) path and exit")
    a = ap.parse_args(argv)
    if a.download:
        print("saved", a.download, "sha256", download_snapshot(SIMPLEQA_URL, REPO_ROOT / a.download))
        return
    if a.download_triviaqa:
        print("saved", a.download_triviaqa, "sha256", download_triviaqa_snapshot(REPO_ROOT / a.download_triviaqa))
        return
    if a.source == TRIVIAQA:
        audit = build_triviaqa(REPO_ROOT / (a.snapshot or TRIVIAQA_SNAPSHOT), a.version, a.exclude or None)
    else:
        audit = build(REPO_ROOT / (a.snapshot or "data/raw/simpleqa_snapshot_2026-09-28.csv"), a.version)
    for c in audit["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print("questions per split:", audit["questions_per_split"])
    sys.exit(0 if all(c["ok"] for c in audit["checks"]) else 1)


if __name__ == "__main__":
    main()
