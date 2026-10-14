"""P1 synthetic-world generator.

    python -m p1_exposure.generate                       # pilot: 120 facts, seed 0
    python -m p1_exposure.generate --n-facts 600 --version main_v0

Outputs (all deterministic for a given seed/config):
    data/synthetic/<version>/   facts.jsonl (holds answers, evaluation-side), train.jsonl,
                                eval.jsonl, idk_train.jsonl, generation_config.json
    data/manifests/synthetic_<version>/   facts_manifest.jsonl (IDs only, no answers),
                                          audit.json
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .names import NamePool
from .relations import DEFAULT_RELATIONS, RELATIONS, render
from .validate import run_checks

REPO_ROOT = Path(__file__).resolve().parent.parent
LEVELS = [0, 1, 2, 5, 10, 20]
IDK_TEXT = "I don't know."
SOURCE = "synthetic"


def build_facts(n_facts, relation_names, levels, dev_frac, seed):
    per_rel_divisor = len(relation_names) * len(levels)
    if n_facts % per_rel_divisor:
        raise SystemExit(f"--n-facts must be divisible by {per_rel_divisor} "
                         f"({len(relation_names)} relations x {len(levels)} levels)")
    n_rel = n_facts // len(relation_names)
    pool = NamePool(seed)
    facts = []
    for rname in relation_names:
        rel = RELATIONS[rname]
        rng = random.Random(f"{seed}-{rname}")
        subjects = [rel.make_subject(pool) for _ in range(n_rel)]
        answers = rel.make_answers(pool, rng, n_rel)
        rel_facts = [{"relation": rname, "subject_type": rel.subject_type,
                      "answer_type": rel.answer_type, "subject": s,
                      "reference_answer": a, "stratum": st}
                     for s, (a, st) in zip(subjects, answers)]

        # Deal exposure levels in blocks of len(levels) over a stratum-sorted order, so
        # every level receives the same mix of answer years / occupations.
        order = list(range(n_rel))
        rng.shuffle(order)
        order.sort(key=lambda i: str(rel_facts[i]["stratum"]))
        for b in range(0, n_rel, len(levels)):
            block_levels = levels[:]
            rng.shuffle(block_levels)
            for i, lvl in zip(order[b:b + len(levels)], block_levels):
                rel_facts[i]["exposure_level"] = lvl

        # Dev / final partition inside every (relation, level) cell.
        cells = defaultdict(list)
        for f in rel_facts:
            cells[f["exposure_level"]].append(f)
        for lvl in levels:
            members = cells[lvl]
            rng.shuffle(members)
            n_dev = round(dev_frac * len(members))
            for k, f in enumerate(members):
                f["split"] = "dev" if k < n_dev else "final"
        facts.extend(rel_facts)

    for n, f in enumerate(facts, start=1):
        f["fact_id"] = f"F{n:04d}"
    return facts, pool


def build_train(facts, seed, version):
    rows = []
    for f in facts:
        e = f["exposure_level"]
        if e == 0:
            continue
        rel = RELATIONS[f["relation"]]
        offset = random.Random(f"{seed}-offset-{f['fact_id']}").randrange(len(rel.train))
        for j in range(e):
            t = (offset + j) % len(rel.train)
            rows.append({
                "presentation_id": f"{f['fact_id']}-p{j + 1:02d}",
                "fact_id": f["fact_id"], "relation": f["relation"],
                "exposure_level": e, "split": f["split"], "presentation_index": j + 1,
                "template_id": f"{f['relation']}.train{t}",
                "prompt": render(rel.train[t], f["subject"]),
                "completion": f["reference_answer"], "dataset_version": version,
            })
    random.Random(f"{seed}-train-order").shuffle(rows)
    return rows


def build_eval(facts, version):
    rows = []
    for f in facts:
        rel = RELATIONS[f["relation"]]
        kind, templates = ("val", rel.val) if f["split"] == "dev" else ("test", rel.test)
        for k, t in enumerate(templates):
            rows.append({
                "question_id": f"{f['fact_id']}-q{k}", "fact_id": f["fact_id"],
                "source_dataset": SOURCE, "dataset_version": version, "split": f["split"],
                "relation": f["relation"], "answer_type": f["answer_type"],
                "exposure_level": f["exposure_level"],
                "template_id": f"{f['relation']}.{kind}{k}",
                "question": render(t, f["subject"]),
                "reference_answer": f["reference_answer"],
            })
    return rows


def build_idk(pool, relation_names, n_per_relation, seed, version):
    rows = []
    n = 0
    for rname in relation_names:
        rel = RELATIONS[rname]
        for k in range(n_per_relation):
            n += 1
            subject = rel.make_subject(pool)
            t = (k + n) % len(rel.train)
            rows.append({
                "idk_id": f"IDK{n:04d}", "relation": rname,
                "template_id": f"{rname}.train{t}",
                "prompt": render(rel.train[t], subject), "subject": subject,
                "completion": IDK_TEXT, "dataset_version": version,
            })
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(n_facts=120, seed=0, version="pilot_v0", relations=None, levels=None,
             dev_frac=0.5, idk_per_relation=6, data_dir=None, manifest_dir=None):
    relations = relations or DEFAULT_RELATIONS
    levels = levels or LEVELS
    data_dir = Path(data_dir or REPO_ROOT / "data" / "synthetic") / version
    manifest_dir = Path(manifest_dir or REPO_ROOT / "data" / "manifests") / f"synthetic_{version}"
    ds_version = f"synthetic-{version}"

    facts, pool = build_facts(n_facts, relations, levels, dev_frac, seed)
    train = build_train(facts, seed, ds_version)
    evals = build_eval(facts, ds_version)
    idk = build_idk(pool, relations, idk_per_relation, seed, ds_version)

    fact_rows = [{k: f[k] for k in ("fact_id", "relation", "subject_type", "answer_type",
                                    "subject", "reference_answer", "exposure_level",
                                    "split", "stratum")} for f in facts]
    manifest_rows = [{k: f[k] for k in ("fact_id", "relation", "exposure_level", "split")}
                     for f in facts]
    config = {"dataset_version": ds_version, "seed": seed, "n_facts": n_facts,
              "relations": relations, "levels": levels, "dev_frac": dev_frac,
              "idk_per_relation": idk_per_relation,
              "templates": {r: {"train": len(RELATIONS[r].train), "val": len(RELATIONS[r].val),
                                "test": len(RELATIONS[r].test)} for r in relations}}

    write_jsonl(data_dir / "facts.jsonl", fact_rows)
    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "eval.jsonl", evals)
    write_jsonl(data_dir / "idk_train.jsonl", idk)
    write_jsonl(manifest_dir / "facts_manifest.jsonl", manifest_rows)
    (data_dir / "generation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")

    checks = run_checks(fact_rows, train, evals, idk, config)
    audit = {
        "dataset_version": ds_version,
        "presentations_per_level": {str(l): sum(1 for r in train if r["exposure_level"] == l)
                                    for l in levels},
        "facts_per_level": dict(sorted(Counter(str(f["exposure_level"]) for f in facts).items(),
                                       key=lambda kv: int(kv[0]))),
        "facts_per_split": dict(Counter(f["split"] for f in facts)),
        "facts_per_relation_level": {r: dict(sorted(Counter(
            str(f["exposure_level"]) for f in facts if f["relation"] == r).items(),
            key=lambda kv: int(kv[0]))) for r in relations},
        "total_train_rows": len(train), "total_eval_rows": len(evals),
        "total_idk_rows": len(idk),
        "checks": checks,
        "file_sha256": {p.name: sha256(p) for p in sorted(data_dir.glob("*.json*"))},
    }
    (manifest_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-facts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default="pilot_v0")
    ap.add_argument("--relations", nargs="+", default=DEFAULT_RELATIONS, choices=list(RELATIONS))
    ap.add_argument("--dev-frac", type=float, default=0.5)
    ap.add_argument("--idk-per-relation", type=int, default=6)
    ap.add_argument("--data-dir")
    ap.add_argument("--manifest-dir")
    a = ap.parse_args(argv)

    audit = generate(a.n_facts, a.seed, a.version, a.relations, LEVELS, a.dev_frac,
                     a.idk_per_relation, a.data_dir, a.manifest_dir)
    for c in audit["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"facts/level: {audit['facts_per_level']}")
    print(f"train presentations/level: {audit['presentations_per_level']}")
    sys.exit(0 if all(c["ok"] for c in audit["checks"]) else 1)


if __name__ == "__main__":
    main()
