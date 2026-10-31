"""P2: run the UNCHANGED base model on a P2 question file (1 greedy + N sampled responses per question),
through the shared code path (common.model.ModelRunner + common.evaluate.run_eval), then grade with common.grading.

    python -m p2_real_qa.generate_responses --split pilot --limit 3          # smoke test: 3 questions, check timing first
    python -m p2_real_qa.generate_responses --split pilot                    # 200 questions -> results/raw/p2_pilot_base.jsonl
    python -m p2_real_qa.generate_responses --split train_probe --device mps

Resumable: rows are appended after every chunk; re-running the same command skips finished question_ids.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import grading, schema                      # noqa: E402
from common.evaluate import run_eval                    # noqa: E402
from common.metrics import behavior_metrics, knowledge_scores  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = "data/real_qa/simpleqa_v0"


def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(requested=None):
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def done_question_ids(path: Path):
    if not path.exists():
        return set()
    return {json.loads(l)["question_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def append_rows(path: Path, rows):
    with open(path, "a", encoding="utf-8", newline="\n") as f:        # same serialisation rule as schema.write_jsonl
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def generate(runner, questions, out_path, model_cfg, dec_cfg, run_id, checkpoint_id, seed, chunk=10, method="base"):
    """Core loop, separated from model loading so tests can pass a fake runner."""
    done = done_question_ids(out_path)
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(done)} questions already done, {len(todo)} to go", file=sys.stderr)
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        rows = run_eval(runner, batch, dec_cfg, model_cfg, checkpoint_id, method, run_id, seed=seed, log_every=0)
        grading.regrade_rows(rows)                     # overwrite the placeholder outcome with the shared grader
        append_rows(out_path, rows)
        el = time.time() - t0
        print(f"  {min(i + chunk, len(todo))}/{len(todo)} questions | {el / (i + len(batch)):.1f}s/question | "
              f"ETA {el / (i + len(batch)) * (len(todo) - i - len(batch)) / 60:.0f} min", file=sys.stderr)


def summarize(path: Path):
    rows = schema.read_jsonl(path)
    greedy = [r for r in rows if r["is_greedy"]]
    print("\n=== greedy behaviour (shared metrics.py) ===")
    print(json.dumps(behavior_metrics([r["outcome"] for r in greedy]), indent=2))
    ks = [v["knowledge_score"] for v in knowledge_scores(rows).values()]
    k0, mid, hi = sum(k == 0 for k in ks), sum(0 < k < 1 for k in ks), sum(k >= 0.8 for k in ks)
    print(f"\n=== K(q) over {len(ks)} questions ===  K=0: {k0} | 0<K<1: {mid} | K>=0.8: {hi}")
    print(f"truncated responses (hit max_new_tokens): {sum(bool(r['truncated']) for r in rows)}/{len(rows)}")
    print(f"invalid responses: {sum(r['outcome'] == 'invalid' for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot", help="pilot | train_probe | validation | locked_test")
    ap.add_argument("--questions", help="override: path to a question JSONL")
    ap.add_argument("--out", help="override output path (default results/raw/p2_<split>_base.jsonl)")
    ap.add_argument("--run-id", help="default p2_<split>_base")
    ap.add_argument("--seed", type=int, default=0, help="base generation seed (shared protocol: greedy=seed, sample i=seed+i+1)")
    ap.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args()

    model_cfg, dec_cfg = load_yaml("model.yaml"), load_yaml("decoding.yaml")
    qpath = Path(a.questions) if a.questions else REPO_ROOT / DATASET_DIR / f"{a.split}.jsonl"
    questions = schema.read_jsonl(qpath)[: a.limit]
    run_id = a.run_id or f"p2_{a.split}_base"
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "raw" / f"p2_{a.split}_base{'_smoke' if a.limit else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    from common.model import ModelRunner, environment_versions   # imported late: needs torch/transformers
    device = pick_device(a.device)
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"], device=device)
    meta = dict(run_id=run_id, device=device, seed=a.seed, model=model_cfg, decoding=dec_cfg,
                questions_file=str(qpath), questions_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
                grading_version=grading.GRADING_VERSION, environment=environment_versions())
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"device={device} dtype={model_cfg['torch_dtype']} | {len(questions)} questions -> {out}", file=sys.stderr)

    generate(runner, questions, out, model_cfg, dec_cfg, run_id, f"{model_cfg['model_id']}-base", a.seed, a.chunk)
    summarize(out)


if __name__ == "__main__":
    main()
