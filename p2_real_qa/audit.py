"""Manual audit of the automatic grades (plan.md P2: "manually audit a sample of automatic grades").

    python -m p2_real_qa.audit make  --responses results/raw/p2_pilot_base.jsonl
        -> results/audit/audit_sheet_<stem>.csv   (open in a spreadsheet, fill human_label + human_notes)
    python -m p2_real_qa.audit score --sheet results/audit/audit_sheet_p2_pilot_base.csv
        -> precision of the grader per predicted class + a file of every grader error

Sampling is STRATIFIED BY PREDICTED CLASS, so per-class precision is estimable but recall is not
(say so in the report). The sheet always includes: every 'correct' (or a random --n-correct of them), every strong
flag, a capped sample of weak hints, and random controls of 'incorrect' and 'abstain'.

    python -m p2_real_qa.audit make --responses results/raw/p2_triviaqa_v0_train_probe_base.jsonl \
        --n-correct 40 --n-weak 40 --n-control 30
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import schema  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID = {"correct", "incorrect", "abstain"}
FIELDS = ["question_id", "sample_index", "is_greedy", "question", "reference_answer", "model_answer",
          "truncated", "auto_outcome", "review_status", "human_label", "human_notes"]


def make(a):
    rows = schema.read_jsonl(a.responses)
    rng = random.Random(7)
    key = lambda r: (r["question_id"], r["sample_index"])
    correct = [r for r in rows if r["outcome"] == "correct"]
    if a.n_correct is not None and a.n_correct < len(correct):        # TriviaQA has hundreds: sample instead of all
        correct = rng.sample(correct, a.n_correct)
    chosen = {key(r): r for r in correct}
    chosen.update({key(r): r for r in rows if r["review_status"] in ("flagged:hedged_attempt", "flagged:numeric_ambiguous")})
    weak = {}
    for r in sorted(rows, key=lambda r: not r["is_greedy"]):                       # one row per question, greedy first
        if r["review_status"] == "flagged:ref_partial_overlap":
            weak.setdefault(r["question_id"], r)
    for r in rng.sample(list(weak.values()), min(a.n_weak, len(weak))):
        chosen[key(r)] = r
    for o in ("incorrect", "abstain"):
        pool = [r for r in rows if r["outcome"] == o and r["review_status"] == "unreviewed"]
        for r in rng.sample(pool, min(a.n_control, len(pool))):
            chosen[key(r)] = r
    out = Path(a.out or REPO_ROOT / "results" / "audit" / f"audit_sheet_{Path(a.responses).stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a byte-order mark so Excel opens accented text correctly ("crème", not "cr√®me")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in chosen.values():
            w.writerow({**{k: r[k] for k in ("question_id", "sample_index", "is_greedy", "question",
                                             "reference_answer", "model_answer", "truncated", "review_status")},
                        "auto_outcome": r["outcome"], "human_label": "", "human_notes": ""})
    print(f"{len(chosen)} rows -> {out}. Fill human_label (correct/incorrect/abstain) and human_notes, then run `score`.")


def read_sheet_text(path) -> str:
    """Reads a filled-in sheet whatever Excel saved it as: UTF-8 with or without a byte-order mark, or (plain
    "CSV" on a Mac) Mac Roman. Labels and IDs are ASCII, so they survive either way."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print(f"note: {path} is not UTF-8 (saved as plain CSV?); reading it as Mac Roman", file=sys.stderr)
        return raw.decode("mac_roman")


def score(a):
    rows = [r for r in csv.DictReader(read_sheet_text(a.sheet).splitlines()) if r["human_label"].strip()]
    if a.responses:      # re-score the SAME human labels against a regraded or regenerated response file
        new = {(r["question_id"], str(r["sample_index"])): r for r in schema.read_jsonl(a.responses)}
        same, changed = [], []
        for r in rows:
            n = new[(r["question_id"], str(r["sample_index"]))]
            # a human label only applies to the text it was given for: if the answer was regenerated with
            # different wording (e.g. a precision change), that row has to be labelled again
            if " ".join(n["model_answer"].split()) != " ".join(r["model_answer"].split()):
                changed.append({**r, "model_answer": n["model_answer"], "auto_outcome": n["outcome"],
                                "review_status": n["review_status"], "human_label": "", "human_notes": ""})
                continue
            r["auto_outcome"], r["review_status"] = n["outcome"], n["review_status"]
            same.append(r)
        rows = same
        print(f"auto outcomes taken from {a.responses} (grading_version {next(iter(new.values()))['grading_version']})")
        if changed:
            relabel = Path(a.sheet).with_name(Path(a.sheet).stem + "__relabel.csv")
            with open(relabel, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(changed)
            print(f"{len(changed)} answers changed wording since they were labelled: left out of this score and "
                  f"written to {relabel}. Label them, paste them into the sheet in place of the old rows, re-run score.")
    bad = [r for r in rows if r["human_label"].strip() not in VALID]
    if bad:
        sys.exit(f"invalid human_label values: {sorted({r['human_label'] for r in bad})}")
    print(f"{len(rows)} labelled rows")
    ct = Counter((r["auto_outcome"], r["human_label"].strip()) for r in rows)
    for (auto, human), c in sorted(ct.items()):
        print(f"  auto={auto:10s} human={human:10s} {c}")
    print("\nPrecision per predicted class:")
    for cls in sorted({r["auto_outcome"] for r in rows}):
        sub = [r for r in rows if r["auto_outcome"] == cls]
        print(f"  {cls:10s} n={len(sub):3d}  human agrees: {sum(r['human_label'].strip() == cls for r in sub) / len(sub):.1%}")
    errs = [r for r in rows if r["human_label"].strip() != r["auto_outcome"]]
    out = Path(a.sheet).with_name("audit_grader_errors.csv")
    # utf-8-sig adds a byte-order mark so Excel opens accented text correctly ("crème", not "cr√®me")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(errs)
    print(f"\n{len(errs)} grader errors -> {out}. Read these, fix common/grading.py, bump GRADING_VERSION, re-run `regrade`.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--responses", required=True)
    m.add_argument("--out")
    m.add_argument("--n-control", dest="n_control", type=int, default=30)
    m.add_argument("--n-weak", dest="n_weak", type=int, default=30)
    m.add_argument("--n-correct", dest="n_correct", type=int, default=None,
                   help="sample this many 'correct' rows instead of including all of them (default: all)")
    s = sp.add_parser("score")
    s.add_argument("--sheet", required=True)
    s.add_argument("--responses", help="optional: score the labels against this (regraded) response file")
    args = ap.parse_args()
    {"make": make, "score": score}[args.cmd](args)
