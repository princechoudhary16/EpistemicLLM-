"""Training questions for real-data abstention fine-tuning (R-Tuning / GeoDe-style baselines).

The questions come from TriviaQA's TRAIN file, which none of the shared splits use (those come from the validation file).
Anything that could leak into evaluation is removed first: every question whose wording, or whose canonical answer
(group), appears in any shared TriviaQA split or in P4's calibration sample.

    python -m p2_abstention.data --download data/raw/triviaqa_rc_nocontext_train_0f7faf33.jsonl     # once (large, not committed)
    python -m p2_abstention.data --snapshot data/raw/triviaqa_rc_nocontext_train_0f7faf33.jsonl     # build the pool

Outputs (deterministic for a given snapshot + seed):
    data/abstention/triviaqa_train_v0/probe.jsonl   D0: labels the knowledge boundary and trains GeoDe's probe
    data/abstention/triviaqa_train_v0/sft.jsonl     D1: the fine-tuning pool every arm selects from
    data/abstention/triviaqa_train_v0/audit.json    counts, exclusions, seeds, hashes, checks
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2_real_qa import data as qa  # noqa: E402
from p2_real_qa.validate import run_checks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "triviaqa_train"                       # question_id = "triviaqa_train-<official id>", never "triviaqa-..."
TRAIN_SNAPSHOT = "data/raw/triviaqa_rc_nocontext_train_0f7faf33.jsonl"
OUT_DIR = "data/abstention/triviaqa_train_v0"
EVAL_SPLITS = ("pilot", "train_probe", "validation", "locked_test")
EVAL_DIR = "data/real_qa/triviaqa_v0"
PROBE_N = 5000
SFT_N = 20000
SEED = 20261107


def download_train_snapshot(path) -> str:
    """Saves the pinned TriviaQA rc.nocontext TRAIN split as a compact JSONL (same row format as the validation snapshot).
    Refuses to overwrite. The parquet shard names are looked up in the pinned revision rather than guessed."""
    from huggingface_hub import HfApi, hf_hub_download

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; pick a new snapshot name instead of overwriting")
    files = sorted(f for f in HfApi().list_repo_files(qa.TRIVIAQA_REPO, repo_type="dataset", revision=qa.TRIVIAQA_REVISION)
                   if f.startswith("rc.nocontext/train-") and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"no rc.nocontext/train-*.parquet in {qa.TRIVIAQA_REPO}@{qa.TRIVIAQA_REVISION}")
    frames = [pd.read_parquet(hf_hub_download(qa.TRIVIAQA_REPO, f, repo_type="dataset", revision=qa.TRIVIAQA_REVISION))
              for f in files]
    df = pd.concat(frames).sort_values("question_id", kind="stable").reset_index(drop=True)
    rows = [dict(question_id=str(r.question_id), question=str(r.question), value=str(r.answer["value"]),
                 aliases=[str(a) for a in r.answer["aliases"]], type=str(r.answer.get("type") or ""))
            for r in df.itertuples()]
    qa.write_jsonl(path, rows)
    print(f"{len(files)} shard(s): {files}", file=sys.stderr)
    return qa.sha256(path)


def eval_exclusions(eval_dir=None, p4_sample=None) -> tuple[set, set]:
    """Normalised question texts and answer groups of every shared evaluation question (all 4 splits + P4's sample)."""
    from common.grading import normalize

    eval_dir = Path(eval_dir or REPO_ROOT / EVAL_DIR)
    texts, groups = set(), set()
    for name in EVAL_SPLITS:
        for r in _read(eval_dir / f"{name}.jsonl"):
            texts.add(qa.normalized_question(r["question"]))
            groups.add(r["group_id"])
    p4 = Path(p4_sample or REPO_ROOT / qa.P4_SAMPLE)
    if p4.exists():
        for r in _read(p4):
            texts.add(qa.normalized_question(r["question"]))
            groups.add("ans:" + (normalize(r["reference_answer"]) or r["question_id"]))
    return texts, groups


def _read(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build(snapshot, data_root=None, eval_dir=None, p4_sample=None, probe_n=PROBE_N, sft_n=SFT_N, seed=SEED) -> dict:
    df, alias_audit = qa.load_triviaqa_snapshot(snapshot)            # dedupes repeated IDs, cleans aliases, groups answers
    texts, groups = eval_exclusions(eval_dir, p4_sample)
    is_text = df.problem.map(qa.normalized_question).isin(texts)
    is_group = df.group_id.isin(groups)
    # pool = df[~(is_text | is_group)].reset_index(drop=True)
    pool = df[~(is_text | is_group)].reset_index(drop=True)
    twin = pool.problem.map(qa.normalized_question).duplicated()     # same question under another ID: keep the first
    pool = pool[~twin].reset_index(drop=True)                        # (df is sorted by ID, so this is deterministic)
    if len(pool) < probe_n + sft_n:
        raise RuntimeError(f"pool has {len(pool)} questions after exclusions; need {probe_n + sft_n}")

    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    picked = {"probe": pool.iloc[sorted(order[:probe_n])], "sft": pool.iloc[sorted(order[probe_n:probe_n + sft_n])]}
    ds_version = f"{SOURCE}-v0"
    records = {name: qa.to_records(d, name, ds_version, SOURCE) for name, d in picked.items()}

    out_dir = Path(data_root or REPO_ROOT / OUT_DIR)
    for name, rows in records.items():
        qa.write_jsonl(out_dir / f"{name}.jsonl", rows)

    checks = [c for c in run_checks(records, {"probe": probe_n, "sft": sft_n})
              if c["name"] != "no_group_shared_across_splits"]       # probe/sft may share answers: both are training data
    leaked = [r["question_id"] for rows in records.values() for r in rows
              if qa.normalized_question(r["question"]) in texts or r["group_id"] in groups]
    checks.append({"name": "no_overlap_with_evaluation", "ok": not leaked,
                   "detail": f"{len(leaked)} questions share wording or answer group with an evaluation split"})
    audit = {
        "dataset_version": ds_version, "snapshot_file": Path(snapshot).name, "snapshot_sha256": qa.sha256(snapshot),
        "source_repo": f"{qa.TRIVIAQA_REPO}@{qa.TRIVIAQA_REVISION}", "seed": seed,
        "unique_questions_in_snapshot": len(df),
        "excluded_same_question_text": int(is_text.sum()), "excluded_same_answer_group": int((is_group & ~is_text).sum()),
        "pool_after_exclusion": len(pool), "questions_per_set": {k: len(v) for k, v in records.items()},
        "dropped_duplicate_question_text": int(twin.sum()),
        "answer_type_per_set": {k: pd.Series([r["answer_type"] for r in v]).value_counts().to_dict()
                                for k, v in records.items()},
        **alias_audit, "checks": checks,
        "file_sha256": {p.name: qa.sha256(p) for p in sorted(out_dir.glob("*.jsonl"))},
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--download", metavar="NEW_SNAPSHOT_PATH", help="download the pinned TriviaQA train split and exit")
    ap.add_argument("--snapshot", default=TRAIN_SNAPSHOT)
    ap.add_argument("--probe-n", type=int, default=PROBE_N)
    ap.add_argument("--sft-n", type=int, default=SFT_N)
    a = ap.parse_args(argv)
    if a.download:
        print("saved", a.download, "sha256", download_train_snapshot(REPO_ROOT / a.download))
        return
    audit = build(REPO_ROOT / a.snapshot, probe_n=a.probe_n, sft_n=a.sft_n)
    for c in audit["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print({k: audit[k] for k in ("unique_questions_in_snapshot", "excluded_same_question_text",
                                 "excluded_same_answer_group", "pool_after_exclusion", "questions_per_set")})
    sys.exit(0 if all(c["ok"] for c in audit["checks"]) else 1)


if __name__ == "__main__":
    main()
