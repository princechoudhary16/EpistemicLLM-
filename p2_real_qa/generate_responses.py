"""P2: run the base model, or a fine-tuned LoRA adapter on top of it, on a P2 question file (1 greedy + N sampled
responses per question), through the shared code path (common.model.ModelRunner + common.evaluate.run_eval), then grade
with common.grading.

    python -m p2_real_qa.generate_responses --split pilot --limit 3          # smoke test: 3 questions, check timing first
    python -m p2_real_qa.generate_responses --split pilot                    # 200 questions -> results/raw/p2_pilot_base.jsonl
    python -m p2_real_qa.generate_responses --split train_probe --device mps
    python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split pilot  # -> results/raw/p2_triviaqa_v0_pilot_base.jsonl

    # a fine-tuned model = the same base model + a LoRA adapter folder (from P5 / P6)
    python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split locked_test \
        --adapter path/to/sft_matched_main_v0_adapter --method sft_matched --training-seed 0
        # -> results/raw/p2_triviaqa_v0_locked_test_sft_matched.jsonl

Resumable: rows are appended after every chunk; re-running the same command skips finished question_ids.
Resuming is refused if the existing file was written by a different model (checked against its .meta.json).
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


def generate(runner, questions, out_path, model_cfg, dec_cfg, run_id, checkpoint_id, seed, chunk=10, method="base",
             training_seed=None):
    """Core loop, separated from model loading so tests can pass a fake runner."""
    done = done_question_ids(out_path)
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(done)} questions already done, {len(todo)} to go", file=sys.stderr)
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        rows = run_eval(runner, batch, dec_cfg, model_cfg, checkpoint_id, method, run_id, seed=seed, training_seed=training_seed, log_every=0)
        grading.regrade_rows(rows)                     # overwrite the placeholder outcome with the shared grader
        append_rows(out_path, rows)
        el = time.time() - t0
        print(f"  {min(i + chunk, len(todo))}/{len(todo)} questions | {el / (i + len(batch)):.1f}s/question | "
              f"ETA {el / (i + len(batch)) * (len(todo) - i - len(batch)) / 60:.0f} min", file=sys.stderr)

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def resolve_model(model_id, adapter=None, method=None, checkpoint_id=None, training_seed=None) -> dict:
    """Which model this run uses and how its rows are labelled. Refuses a label that does not match the weights:
    an adapter needs its own --method, and a method other than "base" needs an adapter."""
    if adapter:
        adapter_path = Path(adapter)
        if not (adapter_path / "adapter_config.json").is_file():
            raise SystemExit(f"--adapter {adapter}: no adapter_config.json there (expected a PEFT/LoRA adapter folder)")
        if not method or method == "base":
            raise SystemExit("--adapter needs --method naming the fine-tuned model (e.g. sft_matched, idk, p5_lam3)")
        return dict(method=method, checkpoint_id=checkpoint_id or f"{method}:{adapter_path.name}",
                    training_seed=training_seed, adapter=str(adapter_path))
    if method and method != "base":
        raise SystemExit(f"--method {method} needs --adapter: without one this would run the base model")
    if training_seed is not None:
        raise SystemExit("--training-seed only makes sense with --adapter (the base model was not trained by us)")
    return dict(method="base", checkpoint_id=checkpoint_id or f"{model_id}-base", training_seed=None, adapter=None)


def adapter_fingerprint(adapter) -> dict:
    """sha256 of the adapter's config and weight files, so every result traces back to the exact weights."""
    if not adapter:
        return {}
    return {name: hashlib.sha256((Path(adapter) / name).read_bytes()).hexdigest()
            for name in ADAPTER_FILES if (Path(adapter) / name).is_file()}


def check_resume(out: Path, model: dict, fingerprint: dict):
    """An output file with rows already in it may only be continued by the SAME model."""
    meta_path = out.with_suffix(".meta.json")
    if not out.exists() or not out.read_text(encoding="utf-8").strip() or not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    old_model = (old.get("checkpoint_id"), old.get("adapter_sha256") or {})
    if old_model[0] is None and model["method"] == "base":
        return                                          # files written before this check existed were base-model runs
    if old_model != (model["checkpoint_id"], fingerprint):
        raise SystemExit(f"{out} already has rows from checkpoint {old_model[0]!r}; refusing to append rows from "
                         f"{model['checkpoint_id']!r}. Use --out for a new file.")


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
    ap.add_argument("--dataset", default=Path(DATASET_DIR).name, help="folder under data/real_qa (simpleqa_v0 | triviaqa_v0)")
    ap.add_argument("--split", default="pilot", help="pilot | train_probe | validation | locked_test")
    ap.add_argument("--questions", help="override: path to a question JSONL")
    ap.add_argument("--out", help="override output path (default results/raw/p2_<split>_<method>.jsonl)")
    ap.add_argument("--run-id", help="default p2_<split>_<method>")
    ap.add_argument("--seed", type=int, default=0, help="base generation seed (shared protocol: greedy=seed, sample i=seed+i+1)")
    ap.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--adapter", help="LoRA adapter folder to load on top of the base model (default: base model only)")
    ap.add_argument("--method", help="method label for the rows and file name: base (default) | sft_matched | idk | p5_lam3 ...")
    ap.add_argument("--checkpoint-id", dest="checkpoint_id", help="exact model label on every row (default: <method>:<adapter folder>)")
    ap.add_argument("--training-seed", dest="training_seed", type=int, help="seed the adapter was trained with (recorded on every row)")
    a = ap.parse_args()

    model_cfg, dec_cfg = load_yaml("model.yaml"), load_yaml("decoding.yaml")
    model = resolve_model(model_cfg["model_id"], a.adapter, a.method, a.checkpoint_id, a.training_seed)
    fingerprint = adapter_fingerprint(model["adapter"])
    qpath = Path(a.questions) if a.questions else REPO_ROOT / "data" / "real_qa" / a.dataset / f"{a.split}.jsonl"
    questions = schema.read_jsonl(qpath)[: a.limit]
    tag = a.split if a.dataset == Path(DATASET_DIR).name else f"{a.dataset}_{a.split}"   # SimpleQA names unchanged
    name = f"p2_{tag}_{model['method']}"                                                 # base runs keep their old names
    run_id = a.run_id or name
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "raw" / f"{name}{'_smoke' if a.limit else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    check_resume(out, model, fingerprint)

    from common.model import ModelRunner, environment_versions   # imported late: needs torch/transformers
    device = pick_device(a.device)
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=model["adapter"],
                         torch_dtype=model_cfg["torch_dtype"], device=device)
    meta = dict(run_id=run_id, device=device, seed=a.seed, model=model_cfg, decoding=dec_cfg,
                method=model["method"], checkpoint_id=model["checkpoint_id"], training_seed=model["training_seed"],
                adapter=model["adapter"], adapter_sha256=fingerprint,
                questions_file=str(qpath), questions_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
                grading_version=grading.GRADING_VERSION, environment=environment_versions())
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"device={device} dtype={model_cfg['torch_dtype']} | model={model['checkpoint_id']} | "
          f"{len(questions)} questions -> {out}", file=sys.stderr)
    
    generate(runner, questions, out, model_cfg, dec_cfg, run_id, model["checkpoint_id"], a.seed, a.chunk,
             method=model["method"], training_seed=model["training_seed"])
    summarize(out)


if __name__ == "__main__":
    main()
