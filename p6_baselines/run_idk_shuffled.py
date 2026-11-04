"""Fixed IDK/abstention baseline (EpistemicLLM Remaining Work and Order, 2026-10-31,
P6 fix: "Rerun the 'I don't know' baseline in float32: shuffle the data, sft_matched
recipe, 2-3 amounts of refusal examples -- it currently refuses 100% of questions").

Root cause of the 100% collapse (p6_training_parity_fix.md, via P1's ablation): the
30 IDK examples were appended after all 3,800 fact presentations with no shuffling,
so the final gradient steps were all refusals. Two changes here:
  1. Shuffle the combined (fact + IDK) training rows -- removes the "last steps are
     all refusals" artifact.
  2. Use the matched recipe (r16, alpha32, dropout0.05, batch4, all 7 linear layers)
     instead of the original under-capacity one -- consistent with sft_matched.
  3. Try a few different *amounts* of IDK presentations (via --idk-repeats, repeating
     the 30 unique IDK entities N times) rather than a single fixed count, since the
     right abstention/fact-learning balance isn't known in advance.

Usage:
    python -m p6_baselines.run_idk_shuffled --dataset-version main_v0 --idk-repeats 1
    python -m p6_baselines.run_idk_shuffled --dataset-version main_v0 --idk-repeats 2
    python -m p6_baselines.run_idk_shuffled --dataset-version main_v0 --idk-repeats 3
"""

import argparse
import random
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

# Same recipe as sft_matched / P1-Aryaman's published recipe.
RECIPE = dict(r=16, alpha=32, dropout=0.05,
              target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
              batch_size=4, lr=2e-4)


def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rows(path):
    return [(r["prompt"], r["completion"]) for r in read_jsonl(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default="main_v0")
    ap.add_argument("--idk-repeats", type=int, default=1,
                     help="repeat the 30 unique IDK entities this many times in the training mix")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--adapter-dir", default="results/final")
    args = ap.parse_args()

    v = args.dataset_version
    data_dir = REPO_ROOT / "data" / "synthetic" / v
    model_cfg = load_yaml("model.yaml")
    dec_cfg = load_yaml("decoding.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_questions = read_jsonl(data_dir / "eval.jsonl")
    fact_rows = load_rows(data_dir / "train.jsonl")
    idk_rows = load_rows(data_dir / "idk_train.jsonl") * args.idk_repeats

    combined = fact_rows + idk_rows
    rng = random.Random(args.seed)
    rng.shuffle(combined)  # the actual fix: no more "last N steps are all refusals"

    method = f"idk_shuffled_r{args.idk_repeats}"
    print(f"dataset={v} | fact rows: {len(fact_rows)} | idk unique: 30 x {args.idk_repeats} repeats "
          f"= {len(idk_rows)} | combined (shuffled): {len(combined)} | method={method}", file=sys.stderr)
    print(f"recipe: r={RECIPE['r']} alpha={RECIPE['alpha']} dropout={RECIPE['dropout']} "
          f"batch={RECIPE['batch_size']} targets={RECIPE['target_modules']} dtype={model_cfg['torch_dtype']}",
          file=sys.stderr)

    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"],
                          torch_dtype=model_cfg["torch_dtype"], device=device)
    lora_cfg = LoraConfig(r=RECIPE["r"], lora_alpha=RECIPE["alpha"], lora_dropout=RECIPE["dropout"],
                           target_modules=list(RECIPE["target_modules"]), task_type="CAUSAL_LM")
    peft_model = get_peft_model(runner.model, lora_cfg).to(device)
    losses, n_trainable = train_lora_sft(
        peft_model, runner.tokenizer, combined, dec_cfg["system_prompt"],
        epochs=args.epochs, lr=RECIPE["lr"], device=device, seed=args.seed, batch_size=RECIPE["batch_size"],
    )
    print(f"  loss first={losses[0]:.4f} last={losses[-1]:.4f} over {len(losses)} steps "
          f"({n_trainable:,} trainable params)", file=sys.stderr)

    adapter_out = REPO_ROOT / args.adapter_dir / f"{method}_{v}_adapter"
    adapter_out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_out))
    del peft_model, runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuned_runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(adapter_out),
                                torch_dtype=model_cfg["torch_dtype"], device=device)
    rows = run_eval(tuned_runner, eval_questions, dec_cfg, model_cfg, f"{method}_v0", method,
                     f"{v}_{method}", seed=args.seed, training_seed=args.seed)
    out_path = REPO_ROOT / args.out_dir / f"p1_{v}_{method}.jsonl"
    write_jsonl(rows, out_path)

    greedy = [r for r in rows if r["is_greedy"]]
    from collections import Counter
    counts = Counter(r["outcome"] for r in greedy)
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    print(f"Greedy outcomes: {dict(counts)} (N={len(greedy)})", file=sys.stderr)


if __name__ == "__main__":
    main()
