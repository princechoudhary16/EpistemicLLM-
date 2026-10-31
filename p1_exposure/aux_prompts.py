"""Answer-masked training-wording prompts for the 0x facts (an auxiliary file for P5's head).

The 0x facts never appear in train.jsonl, so a head trained on train.jsonl alone never sees a 0x
example. This file gives the head those examples without leaking any fact: one prompt per 0x fact
and training-wording template, with NO completion and NO reference answer in the row. The answers stay
in facts.jsonl (evaluation side). Nothing already generated is changed; the file is derived from
facts.jsonl and written to <version>/zero_prompts/.

    python -m p1_exposure.aux_prompts --data-dir data/synthetic --version main_v0

Use these prompts only as auxiliary input with no language-model loss on them. They name the 0x
entities, so a head trained on them has seen those names (never their answers); the 0x level is no
longer entity-novel for that head.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from p1_exposure.relations import RELATIONS, render

FORBIDDEN_KEYS = {"completion", "reference_answer", "answer"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_aux_zero(facts, version):
    rows = []
    for f in facts:
        if f["exposure_level"] != 0:
            continue
        rel = RELATIONS[f["relation"]]
        for t, template in enumerate(rel.train):
            rows.append({
                "aux_id": f"{f['fact_id']}-a{t}", "fact_id": f["fact_id"],
                "relation": f["relation"], "exposure_level": 0, "split": f["split"],
                "template_id": f"{f['relation']}.train{t}",
                "prompt": render(template, f["subject"]),
                "answer_masked": True, "dataset_version": version,
            })
    return rows


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def validate(facts, rows, train_rows, idk_rows, eval_rows):
    out = []
    truth = {f["fact_id"]: f for f in facts}
    zero = [f for f in facts if f["exposure_level"] == 0]

    not_zero = [r["aux_id"] for r in rows if truth.get(r["fact_id"], {}).get("exposure_level") != 0
                or r["exposure_level"] != 0]
    out.append(_check("only_zero_exposure_facts", not not_zero, f"{len(not_zero)} rows for facts with exposure above 0"))

    per_fact = Counter(r["fact_id"] for r in rows)
    wrong = [f["fact_id"] for f in zero if per_fact.get(f["fact_id"], 0) != len(RELATIONS[f["relation"]].train)]
    out.append(_check("every_zero_fact_has_every_train_wording", not wrong,
                      f"{len(zero)} zero-exposure facts; {len(wrong)} with a wrong row count"))

    leaked_keys = [r["aux_id"] for r in rows if FORBIDDEN_KEYS & set(r)]
    out.append(_check("no_answer_field_in_rows", not leaked_keys, f"{len(leaked_keys)} rows carry an answer field"))

    leaked = [r["aux_id"] for r in rows
              if truth[r["fact_id"]]["reference_answer"].lower() in r["prompt"].lower()]
    out.append(_check("answer_text_not_in_prompt", not leaked, f"{len(leaked)} prompts contain their own answer"))

    mismatch = [r["aux_id"] for r in rows
                if r["prompt"] != render(RELATIONS[r["relation"]].train[int(r["template_id"].rsplit("train", 1)[1])],
                                         truth[r["fact_id"]]["subject"])
                or r["relation"] != truth[r["fact_id"]]["relation"] or r["split"] != truth[r["fact_id"]]["split"]]
    out.append(_check("rows_use_train_wording_and_match_fact_table", not mismatch,
                      f"{len(mismatch)} rows differ from the template and fact table"))

    eval_questions = {r["question"] for r in eval_rows}
    replay = [r["aux_id"] for r in rows if r["prompt"] in eval_questions]
    out.append(_check("prompts_not_equal_to_eval_questions", not replay, f"{len(replay)} prompts equal an eval question"))

    corpus = "\n".join([r["prompt"] + "\n" + r["completion"] for r in train_rows]
                       + [r["prompt"] for r in idk_rows])
    seen = sorted({truth[r["fact_id"]]["subject"] for r in rows if truth[r["fact_id"]]["subject"] in corpus})
    out.append(_check("zero_entities_absent_from_train_and_idk", not seen, f"zero-exposure subjects found in training text: {seen[:3]}"))

    ids = [r["aux_id"] for r in rows]
    out.append(_check("ids_unique", len(ids) == len(set(ids)), f"{len(ids)} rows"))
    return out


def run(data_dir, version):
    d = Path(data_dir) / version
    facts = read_jsonl(d / "facts.jsonl")
    rows = build_aux_zero(facts, f"synthetic-{version}")
    idk_path = d / "idk_train.jsonl"
    checks = validate(facts, rows, read_jsonl(d / "train.jsonl"),
                      read_jsonl(idk_path) if idk_path.exists() else [], read_jsonl(d / "eval.jsonl"))
    write_jsonl(d / "zero_prompts" / "aux_zero_prompts.jsonl", rows)
    return checks, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--version", required=True)
    a = ap.parse_args(argv)
    checks, rows = run(a.data_dir, a.version)
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"wrote {len(rows)} prompts to {Path(a.data_dir) / a.version / 'zero_prompts' / 'aux_zero_prompts.jsonl'}")
    sys.exit(0 if all(c["ok"] for c in checks) else 1)


if __name__ == "__main__":
    main()
