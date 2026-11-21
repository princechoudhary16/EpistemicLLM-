"""Float32 re-run of P6's original baselines, ONE phase per job (base | sft | idk).

Same code path as p6_baselines.run_canonical_baselines (its train_and_save and the shared run_eval), same recipe
(LoRA r=8, q_proj+v_proj, lr 2e-4, 1 epoch, seed 0), same data (main_v0). Only the precision differs: it comes from
configs/model.yaml, which is now float32 (team decision 2026-10-31). Split into phases so each fits the 6 h QoS limit
and two can run at once.

    python -m p6_baselines.run_canonical_phase --phase base
    python -m p6_baselines.run_canonical_phase --phase sft
    python -m p6_baselines.run_canonical_phase --phase idk

Writes results/raw/float32/p1_main_v0_<phase>.jsonl (placeholder-graded, as the original script does); regrade with
`python -m p2_real_qa.regrade --responses <file>` to get the v5-graded copy. The bfloat16 files are never touched.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluate import run_eval  # noqa: E402
from common.model import ModelRunner  # noqa: E402
from common.schema import read_jsonl, write_jsonl  # noqa: E402
from p6_baselines.run_canonical_baselines import REPO_ROOT, load_train_rows, load_yaml, train_and_save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "sft", "idk"], required=True)
    ap.add_argument("--dataset-version", default="main_v0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/raw/float32")
    ap.add_argument("--adapter-dir", default="results/final/float32")
    a = ap.parse_args()

    model_cfg, dec_cfg = load_yaml("model.yaml"), load_yaml("decoding.yaml")
    if model_cfg["torch_dtype"] != "float32":
        raise SystemExit(f"configs/model.yaml says {model_cfg['torch_dtype']}; this re-run must be float32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = a.dataset_version
    data_dir = REPO_ROOT / "data" / "synthetic" / v
    questions = read_jsonl(data_dir / "eval.jsonl")
    out_dir = REPO_ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"p1_{v}_{a.phase}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it first if you really want to re-run")

    adapter = None
    if a.phase != "base":
        examples = load_train_rows(data_dir / "train.jsonl")
        if a.phase == "idk":                       # identical to the original: refusal rows appended, unshuffled
            examples = examples + load_train_rows(data_dir / "idk_train.jsonl")
        adapter = REPO_ROOT / a.adapter_dir / f"{a.phase}_{v}_adapter"
        losses, n_trainable = train_and_save(model_cfg, device, examples, dec_cfg["system_prompt"],
                                             a.lora_r, a.lr, a.epochs, a.seed, adapter)
        print(f"{a.phase}: {len(examples)} rows, loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
              f"{len(losses)} steps, {n_trainable} trainable params", file=sys.stderr)

    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(adapter) if adapter else None,
                         torch_dtype=model_cfg["torch_dtype"], device=device)
    ckpt = f"{model_cfg['model_id']}-base" if a.phase == "base" else f"{a.phase}_{v}"
    rows = run_eval(runner, questions, dec_cfg, model_cfg, ckpt, a.phase, f"{v}_{a.phase}", seed=a.seed,
                    training_seed=None if a.phase == "base" else a.seed)
    write_jsonl(rows, out)
    print(f"wrote {len(rows)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
