"""LoRA fine-tuning for one abstention arm (rtuning | geode | random), with the team's matched recipe.

Same recipe as `sft_matched` and P6's IDK baseline (`p6_baselines/run_idk_shuffled.py`): rank 16, alpha 32,
dropout 0.05, all 7 linear layers, batch 4, lr 2e-4, answer-only loss, float32 from configs/model.yaml, data shuffled
with the seed. So arms differ ONLY in their training rows.

    python -m p2_abstention.train --arm rtuning --seed 0
    -> results/final/abstention/rtuning_seed0_adapter/   (+ train_meta.json)

Evaluate with the shared script (validation first; the locked test only after the freeze):
    python -m p2_real_qa.generate_responses --dataset triviaqa_v0 --split validation \
        --adapter results/final/abstention/rtuning_seed0_adapter --method abstention_rtuning --training-seed 0
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SETS_DIR = "data/abstention/triviaqa_train_v0/sets"
RECIPE = dict(r=16, alpha=32, dropout=0.05,
              target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
              batch_size=4, lr=2e-4)


def load_set(path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return [(r["prompt"], r["completion"]) for r in rows]


def shuffled(rows: list, seed: int) -> list:
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def main(argv=None):
    import torch
    import yaml
    from peft import LoraConfig, get_peft_model

    from common.model import ModelRunner, environment_versions
    from common.train import train_lora_sft
    from p2_real_qa.generate_responses import pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("rtuning", "geode", "random"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--sets-dir", default=SETS_DIR)
    ap.add_argument("--adapter-dir", default="results/final/abstention")
    ap.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--limit", type=int, help="smoke test only: first N rows after shuffling")
    a = ap.parse_args(argv)

    model_cfg = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text(encoding="utf-8"))
    dec_cfg = yaml.safe_load((REPO_ROOT / "configs" / "decoding.yaml").read_text(encoding="utf-8"))
    set_path = REPO_ROOT / a.sets_dir / f"{a.arm}.jsonl"
    rows = shuffled(load_set(set_path), a.seed)[: a.limit]
    n_idk = sum(c == "I don't know." for _, c in rows)
    device = pick_device(a.device)
    print(f"arm={a.arm} seed={a.seed} rows={len(rows)} (\"I don't know.\" {n_idk}) device={device} "
          f"dtype={model_cfg['torch_dtype']}", file=sys.stderr)

    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"], device=device)
    lora = LoraConfig(r=RECIPE["r"], lora_alpha=RECIPE["alpha"], lora_dropout=RECIPE["dropout"],
                      target_modules=list(RECIPE["target_modules"]), task_type="CAUSAL_LM")
    model = get_peft_model(runner.model, lora).to(device)
    losses, n_trainable = train_lora_sft(model, runner.tokenizer, rows, dec_cfg["system_prompt"], epochs=a.epochs,
                                         lr=RECIPE["lr"], device=device, seed=a.seed, batch_size=RECIPE["batch_size"])

    name = f"{a.arm}_seed{a.seed}{'_smoke' if a.limit else ''}_adapter"
    out = REPO_ROOT / a.adapter_dir / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    meta = dict(arm=a.arm, seed=a.seed, epochs=a.epochs, rows=len(rows), idk_rows=n_idk, recipe=RECIPE,
                set_file=str(set_path.relative_to(REPO_ROOT)), set_sha256=hashlib.sha256(set_path.read_bytes()).hexdigest(),
                model_id=model_cfg["model_id"], model_revision=model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"],
                prompt_version=dec_cfg["prompt_version"], device=device, trainable_params=n_trainable,
                loss_first=losses[0], loss_last=losses[-1], steps=len(losses),
                loss_mean_last_100=sum(losses[-100:]) / len(losses[-100:]), environment=environment_versions())
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"saved {out} | loss {losses[0]:.3f} -> {losses[-1]:.3f} over {len(losses)} steps", file=sys.stderr)
    del model, runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
