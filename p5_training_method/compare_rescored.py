"""Compare the v5-rescored result files with their pre-regrade copies (regrade_v5.py, Amendment 2 in reports/p5_kq_target.md).

    python p5_training_method/compare_rescored.py [--old results/superseded_containment_grader_final] [--new results/final]

For every p5_*.json present in both folders it flattens both documents and lists
  * every number that changed (path, old, new, difference), largest differences first,
  * every text value that changed (verdicts such as PASS / FAIL, selected lambda, ...), which are the changes that matter most.
.jsonl score files are compared as whole files (changed / unchanged). Nothing is written.
"""
import argparse
import filecmp
import json
from pathlib import Path


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_docs(old, new):
    """Returns (numeric_changes, text_changes, n_values); changes are (path, old, new[, diff])."""
    a, b = dict(flatten(old)), dict(flatten(new))
    nums, texts = [], []
    for path in sorted(set(a) | set(b)):
        x, y = a.get(path, "<missing>"), b.get(path, "<missing>")
        if x == y:
            continue
        both_num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        if both_num:
            nums.append((path, x, y, y - x))
        else:
            texts.append((path, x, y))
    nums.sort(key=lambda t: -abs(t[3]))
    return nums, texts, len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="results/superseded_containment_grader_final")
    ap.add_argument("--new", default="results/final")
    ap.add_argument("--top", type=int, default=4, help="largest numeric changes to show per file")
    args = ap.parse_args()
    old_dir, new_dir = Path(args.old), Path(args.new)

    unchanged, changed_jsonl, changed_json, text_flips = [], [], [], []
    for f in sorted(old_dir.glob("p5_*")):
        g = new_dir / f.name
        if not g.exists() or f.suffix not in (".json", ".jsonl"):
            continue
        if filecmp.cmp(f, g, shallow=False):
            unchanged.append(f.name)
        elif f.suffix == ".jsonl":
            changed_jsonl.append(f.name)
        else:
            changed_json.append(f.name)
    print(f"unchanged (byte-identical): {len(unchanged)}; changed .json: {len(changed_json)}; changed .jsonl: {len(changed_jsonl)}")
    for name in changed_json:
        nums, texts, n = compare_docs(json.loads((old_dir / name).read_text()), json.loads((new_dir / name).read_text()))
        biggest = max((abs(t[3]) for t in nums), default=0.0)
        print(f"\n{name}: {len(nums)} of {n} values changed numerically (largest |diff| {biggest:.4f}), {len(texts)} text changes")
        for path, x, y, d in nums[: args.top]:
            print(f"    {path}: {x:.4f} -> {y:.4f} ({d:+.4f})")
        for path, x, y in texts:
            print(f"    TEXT {path}: {x!r} -> {y!r}")
            text_flips.append((name, path, x, y))
    if changed_jsonl:
        print("\nchanged score files (.jsonl):", ", ".join(changed_jsonl))
    print(f"\ntext values that changed in total: {len(text_flips)}")


if __name__ == "__main__":
    main()
