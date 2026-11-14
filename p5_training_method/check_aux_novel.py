"""Guard for the name-recognition control: prove, from the data, that the 0x entities being scored are novel to the head.

When the 0x facts of --score-split are scored, the head is trained with the aux prompts of the OTHER split only.
This fails (non-zero exit) unless, for the facts being scored:
  - no fact is shared with the aux prompts used for training, and
  - no scored 0x subject name occurs in any training aux prompt, train.jsonl row or idk_train.jsonl row.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--score-split", choices=["dev", "final"], required=True)
    args = ap.parse_args()

    d = REPO / "data" / "synthetic" / args.dataset
    facts = {f["fact_id"]: f for f in read(d / "facts.jsonl")}
    aux = read(d / "zero_prompts" / "aux_zero_prompts.jsonl")
    train_split = "final" if args.score_split == "dev" else "dev"
    train_aux = [r for r in aux if r["split"] == train_split]
    scored = {i: f for i, f in facts.items() if f["exposure_level"] == 0 and f["split"] == args.score_split}

    problems = []
    shared = {r["fact_id"] for r in train_aux} & set(scored)
    if shared:
        problems.append(f"{len(shared)} scored 0x facts also have training aux prompts")
    corpus = "\n".join(
        [r["prompt"] for r in train_aux]
        + [r["prompt"] + " " + r["completion"] for r in read(d / "train.jsonl")]
        + [r["prompt"] + " " + r["completion"] for r in read(d / "idk_train.jsonl")]
    )
    leaked = [f["subject"] for f in scored.values() if f["subject"] in corpus]
    if leaked:
        problems.append(f"{len(leaked)} scored 0x names occur in training text, e.g. {leaked[:3]}")
    print(f"{args.dataset}: scoring {len(scored)} 0x facts of split '{args.score_split}'; head trains on "
          f"{len(train_aux)} aux prompts of {len({r['fact_id'] for r in train_aux})} '{train_split}' facts")
    if problems:
        print("NOT NOVEL: " + "; ".join(problems))
        sys.exit(1)
    print("OK: none of the scored 0x names appears in anything the head trains on")


if __name__ == "__main__":
    main()
