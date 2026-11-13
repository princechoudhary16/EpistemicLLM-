"""Fine-tuning sets for the real-data abstention baselines. Every arm draws from the same SFT pool (D1).

    rtuning   R-Tuning: all of D1. Greedy answer graded correct -> train on the reference answer;
              incorrect or abstained -> train on "I don't know."
    geode     GeoDe (TBG): only the X% of D1 farthest from the probe's hyperplane (largest |d|).
              d > 0 -> reference answer, d < 0 -> "I don't know." (labels from the probe's side, as in the paper)
    random    Control for GeoDe's smaller set: a random X% of D1, labelled like rtuning.
              GeoDe vs random separates "which questions" from "how many questions".

    python -m p2_abstention.build_sets --questions data/abstention/triviaqa_train_v0/sft.jsonl \
        --responses results/raw/p2_abstention_sft_base.jsonl \
        --distances results/abstention/geode_distances.jsonl \
        --out-dir data/abstention/triviaqa_train_v0/sets

Each output row: prompt, completion, question_id, label ("known" | "unknown"), arm.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grading import ALIAS_SEP  # noqa: E402

IDK = "I don't know."
KEEP_FRACTION = 0.25          # GeoDe's X
SEED = 20261107
ARMS = ("rtuning", "geode", "random")


def answer_text(reference_answer: str) -> str:
    """The training target: TriviaQA's canonical answer (the first alias), in natural form. Other aliases are only
    used when they are the SAME answer in different casing ("OLIVER MELLORS" / "Oliver Mellors"), never a related
    name from the list ("Wragby Hall"). Quotes, a trailing full stop and "(labels)" are removed; ALL CAPS is title-cased."""
    from common.grading import normalize

    def clean(a):
        a = re.sub(r"\s+", " ", str(a)).strip().strip("'\"").strip()
        bare = re.sub(r"\s*\([^()]*\)", "", a).strip()
        a = bare or a
        return a[:-1].rstrip() if a.endswith(".") and not a.endswith("..") else a

    aliases = [a for a in str(reference_answer).split(ALIAS_SEP) if a.strip()]
    if not aliases:
        return ""
    canonical = clean(aliases[0])
    same = [clean(a) for a in aliases if normalize(clean(a)) == normalize(canonical)]
    for a in [canonical, *same]:
        if a and not a.isupper():
            return a
    return canonical.title() if canonical.isupper() and len(canonical) > 3 else canonical


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def greedy_outcomes(responses) -> dict:
    return {r["question_id"]: r["outcome"] for r in responses if r["is_greedy"]}


def own_answers(responses) -> dict:
    """question_id -> the base model's own greedy answer, where it was graded correct and not cut off."""
    return {r["question_id"]: r["model_answer"].strip() for r in responses
            if r["is_greedy"] and r["outcome"] == "correct" and not r.get("truncated") and r["model_answer"].strip()}


def build_sets(questions: list, outcomes: dict, distances: dict | None, keep_fraction=KEEP_FRACTION, seed=SEED,
               own: dict | None = None) -> dict:
    """Returns {arm: rows}. Questions whose greedy answer is invalid (empty) are left out of every arm.
    own (from own_answers): known rows are trained on the base model's own correct answer instead of the reference;
    a known row with no own answer (cut off, or geode's probe says known but the greedy answer was wrong) is dropped."""
    usable = [q for q in questions if outcomes.get(q["question_id"]) in ("correct", "incorrect", "abstain")]

    def row(q, known, arm):
        target = answer_text(q["reference_answer"]) if own is None else own.get(q["question_id"])
        return dict(prompt=q["question"], completion=target if known else IDK,
                    question_id=q["question_id"], label="known" if known else "unknown", arm=arm)

    def by_correctness(qs, arm):
        return [row(q, outcomes[q["question_id"]] == "correct", arm) for q in qs]

    n_keep = round(keep_fraction * len(usable))
    sets = {"rtuning": by_correctness(usable, "rtuning")}
    rng = random.Random(seed)
    sets["random"] = by_correctness(sorted(rng.sample(usable, n_keep), key=lambda q: q["question_id"]), "random")
    if distances is not None:
        scored = [q for q in usable if q["question_id"] in distances]
        if len(scored) != len(usable):
            raise ValueError(f"{len(usable) - len(scored)} usable questions have no probe distance")
        far = sorted(scored, key=lambda q: (-abs(distances[q["question_id"]]), q["question_id"]))[:n_keep]
        sets["geode"] = [row(q, distances[q["question_id"]] > 0, "geode")
                         for q in sorted(far, key=lambda q: q["question_id"])]
    if own is not None:
        sets = {arm: [r for r in rows if r["completion"] is not None] for arm, rows in sets.items()}
    for arm, rows in sets.items():
        empty = [r["question_id"] for r in rows if not r["completion"]]
        if empty:
            raise ValueError(f"{arm}: {len(empty)} rows with an empty target (first {empty[:3]})")
    return sets


def summary(sets: dict, outcomes: dict) -> dict:
    out = {}
    for arm, rows in sets.items():
        c = Counter(r["label"] for r in rows)
        agree = sum((r["label"] == "known") == (outcomes[r["question_id"]] == "correct") for r in rows)
        out[arm] = dict(rows=len(rows), known=c["known"], unknown=c["unknown"],
                        unknown_share=round(c["unknown"] / len(rows), 4) if rows else None,
                        label_agrees_with_greedy_correctness=round(agree / len(rows), 4) if rows else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--responses", required=True, help="greedy-graded base-model responses for the same questions")
    ap.add_argument("--distances", help="geode_distances.jsonl from p2_abstention.probe fit (needed for the geode arm)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-fraction", type=float, default=KEEP_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", choices=("reference", "own"), default="reference",
                    help="known-row target: TriviaQA's reference answer, or the base model's own correct answer")
    a = ap.parse_args(argv)

    questions = read_jsonl(a.questions)
    responses = read_jsonl(a.responses)
    outcomes = greedy_outcomes(responses)
    missing = [q["question_id"] for q in questions if q["question_id"] not in outcomes]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no greedy response (first {missing[:3]})")
    distances = ({r["question_id"]: r["distance"] for r in read_jsonl(a.distances)} if a.distances else None)
    # sets = build_sets(questions, outcomes, distances, a.keep_fraction, a.seed)
    own = own_answers(responses) if a.target == "own" else None
    sets = build_sets(questions, outcomes, distances, a.keep_fraction, a.seed, own)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for arm, rows in sets.items():
        with open(out / f"{arm}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = dict(keep_fraction=a.keep_fraction, seed=a.seed, target=a.target, invalid_greedy_left_out=sum(o == "invalid" for o in outcomes.values()),
                inputs={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in filter(None, (a.questions, a.responses, a.distances))},
                arms=summary(sets, outcomes),
                file_sha256={f"{arm}.jsonl": hashlib.sha256((out / f"{arm}.jsonl").read_bytes()).hexdigest() for arm in sets})
    (out / "sets_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info["arms"], indent=2))


if __name__ == "__main__":
    main()
