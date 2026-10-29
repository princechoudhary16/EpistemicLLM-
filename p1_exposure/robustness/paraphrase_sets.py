"""Extra evaluation questions for the paraphrase robustness check (plan.md Week 4, P1).

The standard eval file asks dev facts with the "val" wordings and final facts with the "test"
wordings, so dev vs final mixes two things: which facts, and which wording. Two extra files separate
them. Both leave the existing files untouched and are written to <version>/robustness/.

    heldout  two NEW wordings per relation that appear in no train, val or test template
    swapped  each fact asked with the OTHER split's wording (dev facts get test wordings,
             final facts get val wordings)

    python -m p1_exposure.robustness.paraphrase_sets --data-dir data/synthetic --version main_v0
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from p1_exposure.relations import RELATIONS, render

HELDOUT = {
    "birth_year": ("Tell me the birth year of {s}.", "During which year did {s}'s life begin?"),
    "founder": ("Who is responsible for founding the {s}?", "Tell me who created the {s}."),
    "capital": ("Tell me the capital of {s}.", "The government of {s} sits in which city?"),
    "headquarters": ("Tell me the city where the {s} has its headquarters.",
                     "The headquarters of the {s} is located in which city?"),
    "occupation": ("Tell me what {s} did for work.", "What was the job of {s}?"),
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row(f, qid, template_id, question, version):
    return {"question_id": qid, "fact_id": f["fact_id"], "source_dataset": "synthetic",
            "dataset_version": version, "split": f["split"], "relation": f["relation"],
            "answer_type": f["answer_type"], "exposure_level": f["exposure_level"],
            "template_id": template_id, "question": question, "reference_answer": f["reference_answer"]}


def build_heldout(facts, version):
    rows = []
    for f in facts:
        for k, t in enumerate(HELDOUT[f["relation"]]):
            rows.append(_row(f, f"{f['fact_id']}-h{k}", f"{f['relation']}.heldout{k}",
                             render(t, f["subject"]), version))
    return rows


def build_swapped(facts, version):
    rows = []
    for f in facts:
        rel = RELATIONS[f["relation"]]
        kind, templates = ("test", rel.test) if f["split"] == "dev" else ("val", rel.val)
        for k, t in enumerate(templates):
            rows.append(_row(f, f"{f['fact_id']}-s{k}", f"{f['relation']}.swap_{kind}{k}",
                             render(t, f["subject"]), version))
    return rows


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def validate(facts, heldout, swapped, train_rows, eval_rows):
    out = []
    all_templates = {r: set(RELATIONS[r].train) | set(RELATIONS[r].val) | set(RELATIONS[r].test)
                     for r in RELATIONS}
    overlap = [r for r in HELDOUT if set(HELDOUT[r]) & all_templates[r]]
    out.append(_check("heldout_templates_new", not overlap, f"relations whose held-out wording is not new: {overlap}"))

    train_prompts = {r["prompt"] for r in train_rows}
    eval_questions = {r["question"] for r in eval_rows}
    replay = [r["question_id"] for r in heldout if r["question"] in train_prompts or r["question"] in eval_questions]
    out.append(_check("heldout_not_in_train_or_eval", not replay, f"{len(replay)} held-out questions repeat an existing string"))

    per_fact = {name: Counter(r["fact_id"] for r in rows) for name, rows in (("heldout", heldout), ("swapped", swapped))}
    want = {f["fact_id"]: len(HELDOUT[f["relation"]]) for f in facts}
    bad = [name for name, c in per_fact.items() if any(c.get(fid, 0) != n for fid, n in want.items())]
    out.append(_check("two_questions_per_fact", not bad, f"kinds with a wrong per-fact count: {bad}"))

    truth = {f["fact_id"]: f for f in facts}
    mismatch = [r["question_id"] for r in heldout + swapped
                if r["reference_answer"] != truth[r["fact_id"]]["reference_answer"]
                or r["exposure_level"] != truth[r["fact_id"]]["exposure_level"]
                or r["split"] != truth[r["fact_id"]]["split"]]
    out.append(_check("rows_match_fact_table", not mismatch, f"{len(mismatch)} rows disagree with facts.jsonl"))

    same_wording = [r["question_id"] for r in swapped if r["question"] in eval_questions]
    out.append(_check("swapped_differs_from_original_eval", not same_wording,
                      f"{len(same_wording)} swapped questions equal an original eval question"))

    wrong_set = [r["question_id"] for r in swapped
                 if ("swap_test" in r["template_id"]) != (r["split"] == "dev")]
    out.append(_check("swapped_uses_other_splits_wording", not wrong_set, f"{len(wrong_set)} rows use the wrong wording set"))

    zero = [f for f in facts if f["exposure_level"] == 0]
    covered = {r["fact_id"] for r in heldout} >= {f["fact_id"] for f in zero}
    out.append(_check("zero_exposure_facts_covered", covered, f"{len(zero)} zero-exposure facts"))
    return out


def run(data_dir, version):
    d = Path(data_dir) / version
    facts = read_jsonl(d / "facts.jsonl")
    heldout = build_heldout(facts, f"synthetic-{version}")
    swapped = build_swapped(facts, f"synthetic-{version}")
    checks = validate(facts, heldout, swapped, read_jsonl(d / "train.jsonl"), read_jsonl(d / "eval.jsonl"))
    write_jsonl(d / "robustness" / "eval_heldout.jsonl", heldout)
    write_jsonl(d / "robustness" / "eval_swapped.jsonl", swapped)
    return checks, heldout, swapped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--version", required=True)
    a = ap.parse_args(argv)
    checks, heldout, swapped = run(a.data_dir, a.version)
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"wrote {len(heldout)} held-out and {len(swapped)} swapped questions to "
          f"{Path(a.data_dir) / a.version / 'robustness'}")
    sys.exit(0 if all(c["ok"] for c in checks) else 1)


if __name__ == "__main__":
    main()
