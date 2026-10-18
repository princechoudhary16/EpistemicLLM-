"""P6 Week 1: tiny standard-SFT sanity run (plan.md section 8, Week 1, P6 row:
"run base and tiny standard SFT").

Trains a LoRA adapter on a handful of fictional (question, answer) facts the base
model cannot already know, then proves three things:

  1. Learnability: training actually attaches trainable parameters and the loss
     decreases across steps.
  2. Checkpoint round-trip: the trained adapter can be saved to disk and reloaded
     through the same ModelRunner used everywhere else in this repo -- a *fresh*
     ModelRunner, not the in-memory trained object, so this actually tests the saved
     files rather than papering over a save/load bug.
  3. Behavior change is real: the reloaded model's greedy answers to the training
     questions match what it was taught, distinguishing "the adapter did something"
     from "the run merely completed without crashing."

This is a sanity check, not the canonical standard-SFT baseline: LoRA rank, learning
rate, and target modules here are placeholder defaults (plan.md section 3: "Freeze ...
after toy validation" -- this IS that toy validation, not the frozen result).

Usage:
    python -m p6_baselines.train_sft_smoke
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model import ModelRunner
from common.train import train_lora_sft

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    examples = [(r["question"], r["answer"]) for r in rows]
    # One (question, answer) per distinct fact, in first-seen order, for before/after eval.
    eval_pairs, seen = [], set()
    for q, a in examples:
        if q not in seen:
            eval_pairs.append((q, a))
            seen.add(q)
    return examples, eval_pairs


def show_answers(runner, eval_pairs, system_prompt, max_new_tokens, seed, label):
    print(f"\n=== {label} ===")
    results = []
    for q, a in eval_pairs:
        gen = runner.generate_one(q, system_prompt, max_new_tokens, do_sample=False, seed=seed)
        matched = a.lower() in gen["text"].lower()
        results.append(matched)
        print(f"  Q: {q}\n  expected: {a}\n  model said: {gen['text']!r}\n  matched: {matched}\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="p6_baselines/toy_sft_corpus.jsonl")
    ap.add_argument("--adapter-out", default="results/final/toy_sft_adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_cfg = load_yaml("model.yaml")
    dec_cfg = load_yaml("decoding.yaml")
    examples, eval_pairs = load_corpus(REPO_ROOT / args.corpus)
    print(f"Loaded {len(examples)} training presentations covering {len(eval_pairs)} facts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"], device=device)

    before = show_answers(runner, eval_pairs, dec_cfg["system_prompt"], dec_cfg["max_new_tokens"],
                           args.seed, "BEFORE training (untouched base model)")

    print("=== Training tiny LoRA SFT ===")
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(runner.model, lora_cfg).to(device)
    losses, n_trainable = train_lora_sft(
        peft_model, runner.tokenizer, examples, dec_cfg["system_prompt"],
        epochs=args.epochs, lr=args.lr, device=device, seed=args.seed,
    )
    print(f"Trainable LoRA parameters: {n_trainable}")
    print(f"Loss: first={losses[0]:.4f} last={losses[-1]:.4f} over {len(losses)} steps")
    if losses[-1] >= losses[0]:
        print("WARNING: loss did not decrease -- learnability check failed")

    adapter_out = REPO_ROOT / args.adapter_out
    adapter_out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_out))
    print(f"Saved adapter to {adapter_out}")

    # Drop everything and reload from disk only, so the "after" check exercises the
    # actual saved files, not the in-memory object that was just trained.
    del peft_model, runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuned_runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(adapter_out),
                                torch_dtype=model_cfg["torch_dtype"], device=device)
    after = show_answers(tuned_runner, eval_pairs, dec_cfg["system_prompt"], dec_cfg["max_new_tokens"],
                          args.seed, "AFTER training (fresh reload: base weights + saved adapter from disk)")

    print(f"=== Summary ===\nBefore: {sum(before)}/{len(eval_pairs)} facts already known (expect ~0)")
    print(f"After:  {sum(after)}/{len(eval_pairs)} facts recovered after SFT (expect all {len(eval_pairs)})")


if __name__ == "__main__":
    main()
