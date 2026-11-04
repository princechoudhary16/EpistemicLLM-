"""Regrade P5's saved predictions with the shared v5 grader (common/grading.py, p2_whole_word_v5).

eval_head.py originally graded with its own substring grader ("p5_placeholder_v0"), so every correctness-based P5 number
(accuracy, hallucination, abstention, AUROC / AURC for "predicts correct", K(q) criteria) came from that grader. The team
rule is that every reported number is graded with v5. This script re-scores the SAME saved answers; nothing is generated or
retrained.

    python p5_training_method/regrade_v5.py                 # dry run: report only, no file is changed
    python p5_training_method/regrade_v5.py --apply         # rewrite, keeping the originals

--apply, per run directory <run>/eval_<dataset>_<split>/:
  predictions.jsonl             -> outcome is now the v5 label; outcome_containment keeps the old one; grading_version, review_status added
  predictions_containment.jsonl -> the untouched original (its existence marks the run as already regraded; it is never overwritten)
  summary.json                  -> by_exposure accuracy / hallucination / abstain and auroc_head_predicts_correct recomputed;
                                   the original is kept as summary_containment.json. Spearman and the exposure AUROC do not
                                   depend on grading and are left as they were.
Downstream files (probe_controls output, signal scores, criteria JSONs) are NOT updated here: rerun probe_controls.py and the
criteria scripts afterwards.
"""
import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from common import grading  # noqa: E402


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def regrade_row(row):
    if "outcome_containment" in row:
        raise ValueError("row already regraded")
    outcome, version, status = grading.grade_fields(row["model_answer"], row["reference_answer"], row.get("question", ""))
    return dict(row, outcome=outcome, outcome_containment=row["outcome"], grading_version=version, review_status=status)


def auroc(labels, scores):
    """P(score of a positive > score of a negative), ties count half (same value as sklearn's roc_auc_score)."""
    pos = [s for lab, s in zip(labels, scores) if lab]
    neg = [s for lab, s in zip(labels, scores) if not lab]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def rates(rows, key="outcome"):
    n = len(rows)
    return {o: sum(r[key] == o for r in rows) / n for o in ("correct", "incorrect", "abstain")}


def update_summary(summary, rows):
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["exposure_level"])].append(r)
    for e, rs in groups.items():
        rt = rates(rs)
        summary["by_exposure"][e].update(accuracy=rt["correct"], hallucination=rt["incorrect"], abstain=rt["abstain"])
    summary["auroc_head_predicts_correct"] = auroc([r["outcome"] == "correct" for r in rows], [r["head_pred"] for r in rows])
    summary["grading"] = f"{grading.GRADING_VERSION} (common/grading.py); originals in *_containment files"
    return summary


def regrade_file(pred_path, apply):
    """Returns the per-file statistics; with apply=True also rewrites the files (never overwrites a backup)."""
    pred_path = Path(pred_path)
    backup = pred_path.with_name("predictions_containment.jsonl")
    if backup.exists():
        return {"skipped": "already regraded (predictions_containment.jsonl exists)"}
    rows = read_rows(pred_path)
    new = [regrade_row(r) for r in rows]
    moves = Counter(f"{o['outcome_containment']}>{o['outcome']}" for o in new if o["outcome_containment"] != o["outcome"])
    flags = Counter(o["review_status"] for o in new if o["review_status"] != "unreviewed")
    old_rt, new_rt = rates(new, "outcome_containment"), rates(new)
    stats = {
        "n": len(new), "changed": sum(moves.values()), "moves": dict(moves), "flags": dict(flags),
        "accuracy_containment": old_rt["correct"], "accuracy_v5": new_rt["correct"],
        "hallucination_containment": old_rt["incorrect"], "hallucination_v5": new_rt["incorrect"],
        "abstain_containment": old_rt["abstain"], "abstain_v5": new_rt["abstain"],
        "examples": [{"question": o.get("question"), "answer": o["model_answer"], "reference": o["reference_answer"],
                      "containment": o["outcome_containment"], "v5": o["outcome"]}
                     for o in new if o["outcome_containment"] != o["outcome"]][:5],
    }
    if apply:
        shutil.copy2(pred_path, backup)
        pred_path.write_text("\n".join(json.dumps(r) for r in new), encoding="utf-8")
        summary_path = pred_path.with_name("summary.json")
        if summary_path.exists():
            shutil.copy2(summary_path, pred_path.with_name("summary_containment.json"))
            summary_path.write_text(json.dumps(update_summary(json.loads(summary_path.read_text()), new), indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--apply", action="store_true", help="rewrite predictions.jsonl / summary.json (originals are kept)")
    ap.add_argument("--out", default="results/final/p5_regrade_v5_report.json")
    args = ap.parse_args()

    paths = sorted(Path(args.root).glob("*/eval_*/predictions.jsonl"))
    if not paths:
        sys.exit(f"no predictions found under {args.root}")
    per_file, total_n, total_changed, moves, examples = {}, 0, 0, Counter(), []
    for p in paths:
        st = regrade_file(p, args.apply)
        per_file[str(p.relative_to(args.root))] = st
        if "skipped" in st:
            continue
        total_n += st["n"]
        total_changed += st["changed"]
        moves.update(st["moves"])
        examples += st["examples"][: max(0, 12 - len(examples))]

    done = {k: v for k, v in per_file.items() if "skipped" not in v}
    print(f"{len(paths)} prediction files, {len(done)} regraded in this run, {len(per_file) - len(done)} already done")
    if done:
        print(f"answers: {total_n}; label changes containment -> v5: {total_changed} ({100 * total_changed / total_n:.3f}%)")
        print("moves:", dict(moves))
        worst = max(done.items(), key=lambda kv: abs(kv[1]["accuracy_v5"] - kv[1]["accuracy_containment"]))
        print(f"largest accuracy shift in any file: {worst[1]['accuracy_v5'] - worst[1]['accuracy_containment']:+.4f} ({worst[0]})")
        for e in examples:
            print("  example:", e)
    print("mode:", "APPLIED (originals kept as *_containment files)" if args.apply else "dry run, nothing changed")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"applied": args.apply, "grading_version": grading.GRADING_VERSION, "total_answers": total_n,
                               "total_changed": total_changed, "moves": dict(moves), "per_file": per_file}, indent=2))


if __name__ == "__main__":
    main()
