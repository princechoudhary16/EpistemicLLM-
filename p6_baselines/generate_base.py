"""P6 Week 1 smoke run: reproducible base-model generation + evaluation.

Runs the pinned base model (configs/model.yaml) on a small self-contained question
set with the shared decoding settings (configs/decoding.yaml): one greedy response
plus five sampled responses per question. Writes response-level rows in the common
schema (common/schema.py) and prints a behavior-metrics summary (common/metrics.py).

This does NOT depend on P1/P2 data (plan.md dependency table: P6 needs only "dummy
prediction files + base model" to start) and does NOT implement final grading —
grading here is an explicit placeholder exact/alias/IDK-phrase check
(grading_version="placeholder_exact_v0"), to be replaced by P2's grader per plan.md
section 2 ("P2 maintains shared grading").

Usage:
    python -m p6_baselines.generate_base \
        --fixture p6_baselines/smoke_fixture.jsonl \
        --out results/raw/smoke_base_v0.jsonl \
        --run-id smoke_base_v0 --seed 0
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import schema
from common.metrics import behavior_metrics, knowledge_scores
from common.model import ModelRunner, environment_versions

REPO_ROOT = Path(__file__).resolve().parents[1]
IDK_PHRASES = ("i don't know", "i do not know", "unknown", "cannot determine", "no information")


def placeholder_grade(model_answer, reference_answer):
    """Exact/alias/IDK-phrase check. Placeholder only — see module docstring."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    ans = norm(model_answer)
    if reference_answer == "__unanswerable__":
        return "abstain" if any(p in ans for p in IDK_PHRASES) else "incorrect"
    if any(p in ans for p in IDK_PHRASES):
        return "abstain"
    return "correct" if norm(reference_answer) in ans else "incorrect"


def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint-id", default=None, help="defaults to '<model_id>-base'")
    ap.add_argument("--seed", type=int, default=0, help="base seed; sample i uses seed+i+1, greedy uses seed")
    args = ap.parse_args()

    model_cfg = load_yaml("model.yaml")
    dec_cfg = load_yaml("decoding.yaml")
    checkpoint_id = args.checkpoint_id or f"{model_cfg['model_id']}-base"

    questions = schema.read_jsonl(args.fixture)
    print(f"Loaded {len(questions)} questions from {args.fixture}", file=sys.stderr)
    print("Environment:", json.dumps(environment_versions()), file=sys.stderr)

    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"])

    rows = []
    for q in questions:
        common = dict(
            run_id=args.run_id, question_id=q["question_id"], fact_id=q.get("fact_id"),
            source_dataset=q["source_dataset"], dataset_version=q["dataset_version"], split=q["split"],
            question=q["question"], reference_answer=q["reference_answer"], exposure_level=q.get("exposure_level"),
            model_id=model_cfg["model_id"], model_revision=model_cfg["revision"], checkpoint_id=checkpoint_id,
            method="base", training_seed=None,
            prompt_version=dec_cfg["prompt_version"], decoding_config_id=dec_cfg["decoding_config_id"],
        )

        gens = [("greedy", True, args.seed, dec_cfg["greedy"])]
        for i in range(dec_cfg["sampled"]["n_samples"]):
            gens.append((f"sample{i}", False, args.seed + i + 1, dec_cfg["sampled"]))

        for sample_index, is_greedy, seed, settings in gens:
            gen = runner.generate_one(
                q["question"], dec_cfg["system_prompt"], dec_cfg["max_new_tokens"],
                do_sample=not is_greedy, seed=seed,
                temperature=settings.get("temperature", 1.0), top_p=settings.get("top_p", 1.0),
            )
            outcome = placeholder_grade(gen["text"], q["reference_answer"])
            rows.append(schema.make_row(
                **common,
                generation_seed=seed, sample_index=sample_index, is_greedy=is_greedy,
                model_answer=gen["text"], outcome=outcome, grading_version="placeholder_exact_v0",
                review_status="unreviewed", truncated=gen["truncated"],
                avg_logprob=gen["avg_logprob"], entropy=gen["entropy"],
                confidence_score=gen["avg_logprob"], latency_seconds=gen["latency_seconds"],
            ))

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema.write_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)

    greedy_rows = [r for r in rows if r["is_greedy"]]
    print("\n=== Greedy behavior metrics ===")
    print(json.dumps(behavior_metrics([r["outcome"] for r in greedy_rows]), indent=2))

    print("\n=== Knowledge scores (sampled only) ===")
    for key, ks in sorted(knowledge_scores(rows).items()):
        print(key[2], ks)


if __name__ == "__main__":
    main()
