"""How much does the grader choice matter? Regrades saved predictions with whole-word matching and
compares with the substring (containment) grading used in eval_head.py.

Substring: normalised reference contained anywhere in the normalised answer ("20" would match "2006").
Whole-word: the reference must appear delimited by non-alphanumerics (so "20" does not match "2006").
Abstention (IDK phrase) handling is identical in both. Reads predictions only; changes nothing.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

IDK = ("i dont know", "i do not know", "unknown", "cannot determine", "no information")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def grade(answer, reference, whole_word):
    a, ref = norm(answer), norm(reference)
    if any(p in a for p in IDK):
        return "abstain"
    if whole_word:
        return "correct" if re.search(rf"(?<![a-z0-9]){re.escape(ref)}(?![a-z0-9])", a) else "incorrect"
    return "correct" if ref in a else "incorrect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--dataset", default="main_v0")
    ap.add_argument("--out", default="results/final/p5_regrade_check.json")
    args = ap.parse_args()

    total = Counter()
    per_file = {}
    examples = []
    for path in sorted(Path(args.root).glob(f"{args.dataset}_lam*_seed*/eval_{args.dataset}_*/predictions.jsonl")):
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        n = diff = sub_correct = ww_correct = stored_mismatch = 0
        for r in rows:
            s = grade(r["model_answer"], r["reference_answer"], False)
            w = grade(r["model_answer"], r["reference_answer"], True)
            stored_mismatch += s != r["outcome"]  # sanity: reproduces the stored label
            sub_correct += s == "correct"
            ww_correct += w == "correct"
            if s != w:
                diff += 1
                if len(examples) < 5:
                    examples.append({"answer": r["model_answer"], "reference": r["reference_answer"],
                                     "substring": s, "whole_word": w})
            n += 1
        per_file[str(path.relative_to(args.root))] = {
            "n": n, "label_disagreements": diff, "stored_label_mismatch": stored_mismatch,
            "acc_substring": sub_correct / n, "acc_whole_word": ww_correct / n,
        }
        total.update(n=n, diff=diff, mismatch=stored_mismatch)

    if not per_file:
        raise SystemExit(f"no predictions found under {args.root}")
    print(f"{len(per_file)} prediction files, {total['n']} graded answers")
    print(f"substring vs whole-word label disagreements: {total['diff']} ({100 * total['diff'] / total['n']:.3f}%)")
    print(f"sanity (substring regrade differs from stored label): {total['mismatch']}")
    worst = max(per_file.items(), key=lambda kv: abs(kv[1]["acc_substring"] - kv[1]["acc_whole_word"]))
    print(f"largest accuracy shift in any file: {abs(worst[1]['acc_substring'] - worst[1]['acc_whole_word']):.4f} ({worst[0]})")
    for e in examples:
        print("  example:", e)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": dict(total), "per_file": per_file, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
