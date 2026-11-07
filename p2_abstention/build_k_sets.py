"""K(q)-based abstention arms, declared after the seed-0 validation results (README: "Amendment 1").

K = correct sampled answers / valid sampled answers of the BASE model (5 samples, decoding_v1), the shared
definition (common.metrics.knowledge_scores; the greedy answer is not counted).

    kfilter  K >= 0.8 -> answer;  K == 0 -> "I don't know.";  0 < K < 0.8 dropped    (gray zone removed, by sampling)
    ksoft    answer with probability K, otherwise "I don't know." (seeded draw)       (gray zone kept, graded)
    kthresh  K > p/(1+p) -> answer, otherwise "I don't know."  (p = hallucination penalty; p = 1 -> K > 0.5)

    python -m p2_abstention.build_k_sets --questions data/abstention/triviaqa_train_v0/sft_k8000.jsonl \
        --responses results/raw/p2_abstention_sft_k8000_base_n5.jsonl \
        --out-dir data/abstention/triviaqa_train_v0/sets --match-n 5000
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2_abstention.build_sets import IDK, SEED, answer_text, greedy_outcomes, read_jsonl, summary  # noqa: E402

K_HI = 0.8
PENALTY = 1.0
K_ARMS = ("kfilter", "ksoft", "kthresh")


def k_values(responses) -> dict:
    """question_id -> K, from the sampled rows only (invalid samples left out of the denominator)."""
    from common.metrics import knowledge_scores

    return {key[0]: v["knowledge_score"] for key, v in knowledge_scores(responses, key=("question_id",)).items()}


def target_row(q, known, arm):
    return dict(prompt=q["question"], completion=answer_text(q["reference_answer"]) if known else IDK,
                question_id=q["question_id"], label="known" if known else "unknown", arm=arm)


def build_k_sets(questions, k, hi=K_HI, penalty=PENALTY, seed=SEED, match_n=None) -> dict:
    qs = [q for q in questions if q["question_id"] in k]          # every sample invalid -> no K -> left out
    t = penalty / (1 + penalty)
    draw = random.Random(seed)
    sets = {
        "kfilter": [target_row(q, k[q["question_id"]] >= hi, "kfilter") for q in qs
                    if k[q["question_id"]] >= hi or k[q["question_id"]] == 0],
        "ksoft": [target_row(q, draw.random() < k[q["question_id"]], "ksoft") for q in qs],
        "kthresh": [target_row(q, k[q["question_id"]] > t, "kthresh") for q in qs],
    }
    if match_n:                                                    # same size as geode / random
        pick = random.Random(seed + 1)
        sets = {arm: rows if len(rows) <= match_n else sorted(pick.sample(rows, match_n), key=lambda r: r["question_id"])
                for arm, rows in sets.items()}
    for arm, rows in sets.items():
        empty = [r["question_id"] for r in rows if not r["completion"]]
        if empty:
            raise ValueError(f"{arm}: {len(empty)} rows with an empty target (first {empty[:3]})")
    return sets


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--responses", required=True, help="base-model responses WITH 5 sampled answers (N_SAMPLES=5 label job)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-hi", type=float, default=K_HI)
    ap.add_argument("--penalty", type=float, default=PENALTY)
    ap.add_argument("--match-n", type=int, help="randomly cut each arm to N rows (5000 = the geode/random size)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    questions = read_jsonl(a.questions)
    responses = read_jsonl(a.responses)
    k, outcomes = k_values(responses), greedy_outcomes(responses)
    missing = [q["question_id"] for q in questions if q["question_id"] not in outcomes]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no response yet (first {missing[:3]}): is the label job finished?")
    sets = build_k_sets(questions, k, a.k_hi, a.penalty, a.seed, a.match_n)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for arm, rows in sets.items():
        with open(out / f"{arm}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ks = [k[q["question_id"]] for q in questions if q["question_id"] in k]
    info = dict(k_hi=a.k_hi, penalty=a.penalty, kthresh_answers_if_k_above=a.penalty / (1 + a.penalty),
                match_n=a.match_n, seed=a.seed, questions=len(questions), questions_with_k=len(ks),
                k_buckets=dict(unknown=sum(x == 0 for x in ks), partial=sum(0 < x < a.k_hi for x in ks),
                               known=sum(x >= a.k_hi for x in ks)),
                k_distribution={f"{v:.2f}": n for v, n in sorted(Counter(round(x, 2) for x in ks).items())},
                inputs={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (a.questions, a.responses)},
                arms=summary(sets, outcomes),
                file_sha256={f"{arm}.jsonl": hashlib.sha256((out / f"{arm}.jsonl").read_bytes()).hexdigest() for arm in sets})
    (out / "k_sets_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({x: info[x] for x in ("questions_with_k", "k_buckets", "arms")}, indent=2))


if __name__ == "__main__":
    main()
