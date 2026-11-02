"""Regrade the ablation-harness response files (Colab or cluster runs) with the shared grader.

run_ablation.py writes its own response rows: no question text and its own `outcome`. The shared grader
(common/grading.py) reads the question for some rules, so this script joins the question back from the eval
files and regrades every response row. Original outcomes are kept as `outcome_v1`; nothing else changes.

    python -m p1_exposure.analysis.regrade_colab --run-dir out_v0_s0 --version main_v0 --out-dir regraded/out_v0_s0

A run directory holds responses__<recipe>__<decoding>.jsonl (questions from <version>/eval.jsonl) and, under
extra/<stem>/, the same files for the extra question sets (questions from <version>/robustness/<stem>.jsonl,
for example eval_heldout and eval_swapped). The directory layout is mirrored under --out-dir.

--grader-root: directory whose `common/` package is used instead of this checkout's, for when the grader
version you need is not on this branch yet. --expect-version makes the script fail instead of silently
writing rows graded by a different version.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_EXPECT = "p2_whole_word_v5"


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_grader(grader_root=None, expect_version=DEFAULT_EXPECT):
    if grader_root:
        sys.path.insert(0, str(Path(grader_root).resolve()))
    for name in [m for m in sys.modules if m == "common" or m.startswith("common.")]:
        del sys.modules[name]                       # make sure the requested common/ wins
    from common import grading
    if expect_version and grading.GRADING_VERSION != expect_version:
        raise SystemExit(f"grader is {grading.GRADING_VERSION}, expected {expect_version}; "
                         f"pass --grader-root for a checkout that has it")
    return grading


def question_files(run_dir, data_dir, version):
    """[(responses_path, question_path)] for the main run and every extra question set."""
    run_dir, base = Path(run_dir), Path(data_dir) / version
    out = [(p, base / "eval.jsonl") for p in sorted(run_dir.glob("responses__*.jsonl"))]
    for sub in sorted((run_dir / "extra").glob("*")) if (run_dir / "extra").is_dir() else []:
        out += [(p, base / "robustness" / f"{sub.name}.jsonl") for p in sorted(sub.glob("responses__*.jsonl"))]
    return out


def regrade_file(resp_path, q_path, grading):
    questions = {q["question_id"]: q["question"] for q in read_jsonl(q_path)}
    rows = read_jsonl(resp_path)
    unknown = sorted({r["question_id"] for r in rows} - set(questions))
    if unknown:
        raise SystemExit(f"{resp_path}: {len(unknown)} question_ids are not in {q_path} (first: {unknown[:3]})")
    new = []
    for r in rows:
        r = dict(r)
        r["question"] = questions[r["question_id"]]
        r["outcome_v1"] = r["outcome"]
        new.append(r)
    grading.regrade_rows(new)
    changed = sum(r["outcome"] != r["outcome_v1"] for r in new)
    return new, changed


def run(run_dir, data_dir, version, out_dir, grading):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    summary = []
    for resp, qpath in question_files(run_dir, data_dir, version):
        rows, changed = regrade_file(resp, qpath, grading)
        dest = out_dir / resp.relative_to(run_dir)
        write_jsonl(dest, rows)
        summary.append({"file": str(resp.relative_to(run_dir)).replace("\\", "/"), "rows": len(rows),
                        "changed": changed, "grading_version": grading.GRADING_VERSION})
    if not summary:
        raise SystemExit(f"no responses__*.jsonl files found in {run_dir}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--grader-root")
    ap.add_argument("--expect-version", default=DEFAULT_EXPECT)
    a = ap.parse_args(argv)
    grading = load_grader(a.grader_root, a.expect_version)
    for s in run(a.run_dir, a.data_dir, a.version, a.out_dir, grading):
        print(f"{s['file']}: {s['rows']} rows, {s['changed']} changed ({s['grading_version']})")


if __name__ == "__main__":
    main()
