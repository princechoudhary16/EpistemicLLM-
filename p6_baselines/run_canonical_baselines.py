"""P6 Weeks 2-3: canonical base / standard-SFT / IDK baseline runs (plan.md section 8:
Week 2 P6 row "produce pilot base/SFT/IDK predictions"; Week 3 P6 row "complete
canonical SFT/IDK runs").

Trains ONE canonical standard-SFT checkpoint and ONE IDK/abstention checkpoint, both
from the identical base checkpoint, LoRA config, learning rate, epochs and seed --
plan.md section 7's training-parity requirement. The only intentional difference is
IDK's extra unknown-entity examples, and that difference is logged explicitly (exact
counts printed at train time) rather than left implicit.

Data: P1's synthetic dataset, selected via --dataset-version (checked out from
origin/p1/synthetic-generator -- see git log for the exact commit/generator run).
Defaults to pilot_v0 (120 facts); pass --dataset-version main_v0 for the full
600-fact grid once generated.

Usage:
    python -m p6_baselines.run_canonical_baselines
    python -m p6_baselines.run_canonical_baselines --dataset-version main_v0
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


def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_rows(path):
    return [(r["prompt"], r["completion"]) for r in read_jsonl(path)]


def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_and_save(model_cfg, device, examples, system_prompt, lora_r, lr, epochs, seed, adapter_out):
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"],
                          torch_dtype=model_cfg["torch_dtype"], device=device)
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.0,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(runner.model, lora_cfg).to(device)
    losses, n_trainable = train_lora_sft(peft_model, runner.tokenizer, examples, system_prompt,
                                          epochs=epochs, lr=lr, device=device, seed=seed)
    adapter_out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_out))
    del peft_model, runner
    free_gpu()
    return losses, n_trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--adapter-dir", default="results/final")
    ap.add_argument("--dataset-version", default="pilot_v0")
    args = ap.parse_args()

    model_cfg = load_yaml("model.yaml")
    dec_cfg = load_yaml("decoding.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = REPO_ROOT / "data" / "synthetic" / args.dataset_version
    v = args.dataset_version

    eval_questions = read_jsonl(data_dir / "eval.jsonl")
    train_examples = load_train_rows(data_dir / "train.jsonl")
    idk_examples = load_train_rows(data_dir / "idk_train.jsonl")
    print(f"eval: {len(eval_questions)} questions | train: {len(train_examples)} presentations "
          f"| idk: {len(idk_examples)} extra examples", file=sys.stderr)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = REPO_ROOT / args.adapter_dir

    # 1) base (untouched model)
    print("=== [1/3] base model eval ===", file=sys.stderr)
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"],
                          torch_dtype=model_cfg["torch_dtype"], device=device)
    base_rows = run_eval(runner, eval_questions, dec_cfg, model_cfg, f"{model_cfg['model_id']}-base",
                          "base", f"{v}_base", seed=args.seed)
    write_jsonl(base_rows, out_dir / f"p1_{v}_base.jsonl")
    del runner
    free_gpu()

    # 2) standard SFT -- trained on train.jsonl only
    print("=== [2/3] standard SFT: train + eval ===", file=sys.stderr)
    sft_adapter = adapter_dir / f"sft_{v}_adapter"
    losses, n_trainable = train_and_save(model_cfg, device, train_examples, dec_cfg["system_prompt"],
                                          args.lora_r, args.lr, args.epochs, args.seed, sft_adapter)
    print(f"  SFT: {len(train_examples)} presentations, loss first={losses[0]:.4f} last={losses[-1]:.4f} "
          f"over {len(losses)} steps ({n_trainable} trainable params)", file=sys.stderr)
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(sft_adapter),
                          torch_dtype=model_cfg["torch_dtype"], device=device)
    sft_rows = run_eval(runner, eval_questions, dec_cfg, model_cfg, f"sft_{v}", "sft", f"{v}_sft",
                         seed=args.seed, training_seed=args.seed)
    write_jsonl(sft_rows, out_dir / f"p1_{v}_sft.jsonl")
    del runner
    free_gpu()

    # 3) IDK/abstention -- trained on train.jsonl + idk_train.jsonl (extra examples disclosed below)
    print("=== [3/3] IDK baseline: train + eval ===", file=sys.stderr)
    idk_adapter = adapter_dir / f"idk_{v}_adapter"
    idk_combined = train_examples + idk_examples
    losses, n_trainable = train_and_save(model_cfg, device, idk_combined, dec_cfg["system_prompt"],
                                          args.lora_r, args.lr, args.epochs, args.seed, idk_adapter)
    print(f"  IDK: {len(idk_combined)} presentations ({len(train_examples)} fact + {len(idk_examples)} IDK -- "
          f"+{len(idk_examples)} extra vs standard SFT), loss first={losses[0]:.4f} last={losses[-1]:.4f} "
          f"over {len(losses)} steps ({n_trainable} trainable params)", file=sys.stderr)
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(idk_adapter),
                          torch_dtype=model_cfg["torch_dtype"], device=device)
    idk_rows = run_eval(runner, eval_questions, dec_cfg, model_cfg, f"idk_{v}", "idk", f"{v}_idk",
                         seed=args.seed, training_seed=args.seed)
    write_jsonl(idk_rows, out_dir / f"p1_{v}_idk.jsonl")

    print("=== done ===", file=sys.stderr)
    for name in ("base", "sft", "idk"):
        print(f"  {out_dir / f'p1_{v}_{name}.jsonl'}", file=sys.stderr)


if __name__ == "__main__":
    main()
