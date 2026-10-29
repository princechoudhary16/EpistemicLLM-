"""Standard-SFT baseline trained with a LoRA recipe that matches a specific other
run, for a training-parity comparison (plan.md section 7: "Same initial model, LoRA
configuration, data order, and exposure budget where methods are comparable" -- a
joint P5+P6 requirement).

Context: P6's canonical SFT baseline (run_canonical_baselines.py) uses r=8,
target_modules=(q_proj, v_proj), batch=1 -- deliberately minimal, per plan.md section
3's "start with ordinary LoRA" guidance. A teammate review correctly pointed out that
comparing an auxiliary-head-and-loss run against that baseline isn't valid unless the
plain-SFT side uses the exact same LoRA recipe as the auxiliary-loss side. Absent
that exact recipe, this reuses P1/Aryaman's own published recipe ("mine" in
p1_exposure/ablation/run_ablation.py) as the best available matched point: r=16,
alpha=32, dropout=0.05, target_modules=all 7 linear layers, batch=4, lr=2e-4.

This produces an ADDITIONAL method ("sft_matched"), not a replacement for the
original canonical "sft" baseline -- both stay available, clearly labeled, so
nothing already reported becomes ambiguous.

Usage:
    python -m p6_baselines.run_matched_sft --dataset-version pilot_v0
    python -m p6_baselines.run_matched_sft --dataset-version main_v0
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluate import run_eval
from common.model import ModelRunner
from common.schema import read_jsonl, write_jsonl
from common.train import train_lora_sft

REPO_ROOT = Path(__file__).resolve().parents[1]

# P1/Aryaman's "mine" recipe, exactly as defined in p1_exposure/ablation/run_ablation.py.
ARYAMAN_RECIPE = dict(
    r=16, alpha=32, dropout=0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    batch_size=4, lr=2e-4,
)


def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_rows(path):
    return [(r["prompt"], r["completion"]) for r in read_jsonl(path)]


def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default="pilot_v0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--adapter-dir", default="results/final")
    args = ap.parse_args()

    recipe = ARYAMAN_RECIPE
    v = args.dataset_version
    data_dir = REPO_ROOT / "data" / "synthetic" / v
    model_cfg = load_yaml("model.yaml")
    dec_cfg = load_yaml("decoding.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_questions = read_jsonl(data_dir / "eval.jsonl")
    train_examples = load_train_rows(data_dir / "train.jsonl")
    print(f"dataset={v} | eval: {len(eval_questions)} questions | train: {len(train_examples)} presentations",
          file=sys.stderr)
    print(f"recipe: r={recipe['r']} alpha={recipe['alpha']} dropout={recipe['dropout']} "
          f"batch={recipe['batch_size']} targets={recipe['target_modules']}", file=sys.stderr)

    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"],
                          torch_dtype=model_cfg["torch_dtype"], device=device)
    lora_cfg = LoraConfig(r=recipe["r"], lora_alpha=recipe["alpha"], lora_dropout=recipe["dropout"],
                           target_modules=list(recipe["target_modules"]), task_type="CAUSAL_LM")
    peft_model = get_peft_model(runner.model, lora_cfg).to(device)
    losses, n_trainable = train_lora_sft(
        peft_model, runner.tokenizer, train_examples, dec_cfg["system_prompt"],
        epochs=args.epochs, lr=recipe["lr"], device=device, seed=args.seed, batch_size=recipe["batch_size"],
    )
    print(f"  loss first={losses[0]:.4f} last={losses[-1]:.4f} over {len(losses)} steps "
          f"({n_trainable:,} trainable params)", file=sys.stderr)

    adapter_out = REPO_ROOT / args.adapter_dir / f"sft_matched_{v}_adapter"
    adapter_out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_out))
    del peft_model, runner
    free_gpu()

    tuned_runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(adapter_out),
                                torch_dtype=model_cfg["torch_dtype"], device=device)
    rows = run_eval(tuned_runner, eval_questions, dec_cfg, model_cfg, "sft_matched_v0", "sft_matched",
                     f"{v}_sft_matched", seed=args.seed, training_seed=args.seed)
    out_path = REPO_ROOT / args.out_dir / f"p1_{v}_sft_matched.jsonl"
    write_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
