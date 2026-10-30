"""Teacher K labels for the K(q)-target experiment (reports/p5_kq_target.md).

A fixed teacher (the lambda=0 seed-0 checkpoint of a world, plain SFT) answers P1's HELD-OUT wordings
(data/synthetic/<world>/robustness/eval_heldout.jsonl: 2 questions per fact for all 600 facts, wordings in no train, validation
or test template) through P6's ModelRunner and run_eval (decoding config, placeholder grader). K(fact) is the project's K(q)
(common.metrics.knowledge_scores, used directly): correct sampled responses / sampled responses, greedy excluded, here out of
10 per fact (2 wordings x 5 samples). Writes the schema response file and a small labels JSON with full provenance.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from common.evaluate import run_eval  # noqa: E402
from common.metrics import knowledge_scores  # noqa: E402
from common.model import ModelRunner  # noqa: E402
from common.schema import read_jsonl, write_jsonl  # noqa: E402


def compute_k(rows):
    """{fact_id: {"knowledge_score": K, "n_samples": n}} from schema response rows (P6's definition, per fact)."""
    scores = knowledge_scores(rows, key=("run_id", "checkpoint_id", "fact_id"))
    return {group[2]: v for group, v in scores.items()}


def summarise(k_by_fact, exposure_by_fact):
    by_level = defaultdict(list)
    for fact, v in k_by_fact.items():
        by_level[exposure_by_fact[fact]].append(v["knowledge_score"])
    exposed = [v["knowledge_score"] for f, v in k_by_fact.items() if exposure_by_fact[f] > 0]
    return {
        "k_mean_by_exposure": {str(e): sum(v) / len(v) for e, v in sorted(by_level.items())},
        "share_exposed_facts_with_k_between_0.1_and_0.9": sum(0.1 <= k <= 0.9 for k in exposed) / len(exposed),
        "share_exposed_facts_with_k_0": sum(k == 0 for k in exposed) / len(exposed),
        "share_exposed_facts_with_k_1": sum(k == 1 for k in exposed) / len(exposed),
    }


def load_yaml(name):
    with open(REPO / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default="main_v0")
    ap.add_argument("--teacher-lam", default="0")
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    args = ap.parse_args()

    ds = args.dataset_version
    model_cfg, dec_cfg = load_yaml("model.yaml"), load_yaml("decoding.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = REPO / "data" / "synthetic" / ds
    heldout_path = data_dir / "robustness" / "eval_heldout.jsonl"
    questions = read_jsonl(heldout_path)[: args.limit]
    exposure_by_fact = {f["fact_id"]: f["exposure_level"] for f in read_jsonl(data_dir / "facts.jsonl")}

    teacher_run = f"{ds}_lam{args.teacher_lam}_seed{args.teacher_seed}"
    ckpt = Path(args.root) / teacher_run / "checkpoint"
    print(f"teacher {ckpt}; {len(questions)} held-out questions; decoding={dec_cfg['decoding_config_id']}",
          file=sys.stderr, flush=True)
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(ckpt / "adapter"),
                         torch_dtype=model_cfg["torch_dtype"], device=device)
    rows = run_eval(runner, questions, dec_cfg, model_cfg, f"kq_teacher_{teacher_run}", "kq_teacher",
                    f"{ds}_kq_teacher_heldout", seed=0, training_seed=args.teacher_seed)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out_dir / f"p5_kq_teacher_{ds}_heldout.jsonl")

    k_by_fact = compute_k(rows)
    if args.limit is None:
        assert len(k_by_fact) == len(exposure_by_fact), "every fact must be labelled"
        assert {v["n_samples"] for v in k_by_fact.values()} == {10}, "expected 2 wordings x 5 sampled responses per fact"
    doc = {
        "dataset": ds,
        "teacher": {"run": teacher_run, "lambda": args.teacher_lam, "training_seed": args.teacher_seed},
        "labelled_questions": "robustness/eval_heldout.jsonl (wordings in no train, validation or test template)",
        "heldout_file_sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
        "decoding_config_id": dec_cfg["decoding_config_id"],
        "prompt_version": dec_cfg["prompt_version"],
        "grading_version": rows[0]["grading_version"],
        "k_definition": "common.metrics.knowledge_scores: correct sampled responses / sampled responses, greedy excluded",
        "n_sampled_responses_per_fact": 10,
        "n_generations_total": len(rows),
        "n_facts": len(k_by_fact),
        **summarise(k_by_fact, exposure_by_fact),
        "labels": {f: v["knowledge_score"] for f, v in sorted(k_by_fact.items())},
    }
    labels_out = Path(args.labels_out or REPO / "results" / "final" / f"p5_kq_labels_{ds}.json")
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text(json.dumps(doc, indent=2))
    print(json.dumps({k: v for k, v in doc.items() if k != "labels"}, indent=2))
    print(f"wrote {labels_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
