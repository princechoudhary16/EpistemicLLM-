"""Integrity checks for a generated synthetic world.

    python -m p1_exposure.validate data/synthetic/pilot_v0
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .relations import RELATIONS, render

NAME_ANSWER_TYPES = {"person", "place"}  # answers that are unique invented names


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_checks(facts, train, evals, idk, config):
    levels = config["levels"]
    relations = config["relations"]
    out = []

    # 1. every exposure level present and balanced, overall and per relation
    by_level = Counter(f["exposure_level"] for f in facts)
    cell = Counter((f["relation"], f["exposure_level"]) for f in facts)
    balanced = (set(by_level) == set(levels) and len(set(by_level.values())) == 1
                and all(cell[(r, l)] == len(facts) // (len(relations) * len(levels))
                        for r in relations for l in levels))
    out.append(_check("exposure_groups_balanced", balanced,
                      f"facts per level {dict(sorted(by_level.items()))}"))

    # 2. dev/final split populated in every level
    split_cell = Counter((f["exposure_level"], f["split"]) for f in facts)
    missing = [(l, s) for l in levels for s in ("dev", "final") if split_cell[(l, s)] == 0]
    out.append(_check("splits_cover_all_levels", not missing, f"empty (level, split): {missing}"))

    # 3. presentations in the training corpus equal the assigned exposure
    presented = Counter(r["fact_id"] for r in train)
    bad = [f["fact_id"] for f in facts if presented.get(f["fact_id"], 0) != f["exposure_level"]]
    out.append(_check("presentations_equal_exposure", not bad,
                      f"{len(bad)} facts with a mismatch" if bad else "every fact matches"))

    # 4. no 0x fact leaks into any training text
    zero = [f for f in facts if f["exposure_level"] == 0]
    train_text = [r["prompt"].lower() + " " + r["completion"].lower() for r in train + idk]
    leaks = []
    for f in zero:
        needles = [f["subject"].lower()]
        if f["answer_type"] in NAME_ANSWER_TYPES:
            needles.append(f["reference_answer"].lower())
        if any(n in t for n in needles for t in train_text):
            leaks.append(f["fact_id"])
    zero_ids = {f["fact_id"] for f in zero}
    leaks += [r["fact_id"] for r in train if r["fact_id"] in zero_ids]
    out.append(_check("no_zero_exposure_in_training", not leaks,
                      f"{len(zero)} zero-exposure facts checked, leaks: {sorted(set(leaks))}"))

    # 5. unique names everywhere; no subject reused across facts
    names = [f["subject"].lower() for f in facts] + [r["subject"].lower() for r in idk]
    names += [f["reference_answer"].lower() for f in facts if f["answer_type"] in NAME_ANSWER_TYPES]
    dup_names = [n for n, c in Counter(names).items() if c > 1]
    out.append(_check("names_unique_across_world", not dup_names, f"duplicates: {dup_names[:5]}"))
    dup_rel = [k for k, c in Counter((f["subject"], f["relation"]) for f in facts).items() if c > 1]
    out.append(_check("no_duplicate_subject_relation", not dup_rel, f"duplicates: {dup_rel[:5]}"))

    # 6. contradictions: one prompt must never map to two completions
    completions = defaultdict(set)
    for r in train + idk:
        completions[r["prompt"]].add(r["completion"])
    contradictions = [p for p, c in completions.items() if len(c) > 1]
    out.append(_check("no_contradictory_prompts", not contradictions,
                      f"{len(contradictions)} prompts with conflicting completions"))

    # 7. evaluation wording never appears in training
    template_overlap = []
    for r in relations:
        rel = RELATIONS[r]
        if set(rel.train) & (set(rel.val) | set(rel.test)) or set(rel.val) & set(rel.test):
            template_overlap.append(r)
    train_prompts = {r["prompt"] for r in train + idk}
    replayed = [e["question_id"] for e in evals if e["question"] in train_prompts]
    out.append(_check("eval_templates_disjoint_from_training",
                      not template_overlap and not replayed,
                      f"template overlap: {template_overlap}; replayed questions: {len(replayed)}"))

    # 8. dev facts use validation wording only, final facts use test wording only
    wrong = [e["question_id"] for e in evals
             if e["template_id"].split(".")[1][:-1] != ("val" if e["split"] == "dev" else "test")]
    per_fact = Counter(e["fact_id"] for e in evals)
    expected = {f["fact_id"]: len(RELATIONS[f["relation"]].val) for f in facts}
    short = [fid for fid, n in expected.items() if per_fact.get(fid, 0) != n]
    out.append(_check("eval_split_and_coverage", not wrong and not short,
                      f"wrong-template rows: {len(wrong)}; facts missing questions: {len(short)}"))

    # 9. eval rows agree with the fact table
    truth = {f["fact_id"]: f for f in facts}
    mismatch = [e["question_id"] for e in evals
                if e["reference_answer"] != truth[e["fact_id"]]["reference_answer"]
                or e["exposure_level"] != truth[e["fact_id"]]["exposure_level"]
                or e["question"] not in {
                    render(t, truth[e["fact_id"]]["subject"])
                    for t in RELATIONS[e["relation"]].val + RELATIONS[e["relation"]].test}]
    out.append(_check("eval_rows_match_facts", not mismatch, f"{len(mismatch)} inconsistent rows"))

    # 10. answer mix per level (year decade / occupation) not skewed by exposure
    skew = []
    for r in relations:
        strata = defaultdict(Counter)
        for f in facts:
            if f["relation"] == r and f["stratum"] is not None:
                strata[f["stratum"]][f["exposure_level"]] += 1
        for s, counts in strata.items():
            per_level = [counts.get(l, 0) for l in levels]
            if max(per_level) - min(per_level) > 2:
                skew.append((r, s))
    out.append(_check("answer_mix_balanced_across_levels", not skew, f"skewed strata: {skew[:5]}"))
    return out


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None):
    d = Path((argv or sys.argv[1:])[0])
    config = json.loads((d / "generation_config.json").read_text(encoding="utf-8"))
    checks = run_checks(_load(d / "facts.jsonl"), _load(d / "train.jsonl"),
                        _load(d / "eval.jsonl"), _load(d / "idk_train.jsonl"), config)
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    sys.exit(0 if all(c["ok"] for c in checks) else 1)


if __name__ == "__main__":
    main()
