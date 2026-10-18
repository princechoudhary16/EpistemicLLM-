import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import peft
import torch
import transformers
import yaml

from data import collate, encode_row, load_rows
from model import build_model, save_checkpoint

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha256():
    h = hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out-root", default=str(REPO / "results" / "raw" / "p5"))
    ap.add_argument("--allow-non-cuda", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.lam is not None:
        cfg["head"]["aux_weight"] = args.lam
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    assert cfg["train"]["epochs"] == 1, "exposure counts assume exactly one pass over the data"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_non_cuda:
        raise SystemExit("No CUDA device found. Run on the GPU node, or pass --allow-non-cuda for local debugging.")

    data_path = Path(args.data) if args.data else REPO / cfg["data"]["train_path"]
    torch.manual_seed(cfg["train"]["seed"])
    tokenizer, model = build_model(cfg, device)

    system_prompt = cfg["system_prompt"]
    rows = load_rows(data_path)[: args.limit]
    examples = [
        encode_row(tokenizer, system_prompt, r, cfg["data"]["max_seq_len"], cfg["head"]["max_exposure"])
        for r in rows
    ]
    bs = cfg["train"]["batch_size"]
    batches = [collate(examples[i : i + bs], tokenizer.pad_token_id) for i in range(0, len(examples), bs)]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"])

    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    aux_weight = cfg["head"]["aux_weight"]
    run_config = {
        "config": cfg,
        "run_id": args.run_id,
        "code_sha256": code_sha256(),
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "n_rows": len(examples),
        "n_steps": len(batches),
        "n_trainable_params": sum(p.numel() for p in params),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none",
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    meta = {
        "aux_weight": aux_weight,
        "target": "log(1+e)/log(1+max_exposure)",
        "head_layer": "last hidden layer, after final norm",
        "pooling": "last prompt token",
        "config": cfg,
    }

    model.train()
    metrics = open(out_dir / "metrics.jsonl", "w")
    wall_start = time.perf_counter()
    total_tokens = 0
    for step, batch in enumerate(batches, 1):
        batch = to_device(batch, device)
        sync(device)
        t0 = time.perf_counter()
        out = model(batch, aux_weight)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        sync(device)
        dt = time.perf_counter() - t0

        tokens = int(batch["attention_mask"].sum())
        total_tokens += tokens
        record = {
            "step": step,
            "loss": out["loss"].item(),
            "l_answer": out["l_answer"].item(),
            "l_aux": out["l_aux"].item(),
            "pred_mean": out["pred"].mean().item(),
            "step_seconds": dt,
            "tokens": tokens,
        }
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()
        if step % args.log_every == 0 or step == len(batches):
            print(
                f"step {step}/{len(batches)} | l_answer {record['l_answer']:.4f} | "
                f"l_aux {record['l_aux']:.4f} | pred_mean {record['pred_mean']:.3f} | {dt:.2f}s",
                flush=True,
            )
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(model, out_dir / "checkpoint_latest", meta)
    wall = time.perf_counter() - wall_start
    metrics.close()

    model.eval()
    by_exposure = collections.defaultdict(list)
    sq_err = []
    with torch.no_grad():
        for b in batches:
            b = to_device(b, device)
            pred = model(b, 0.0)["pred"].cpu()
            target = b["s_target"].cpu()
            sq_err += ((pred - target) ** 2).tolist()
            for e, p in zip(b["exposure"].cpu().tolist(), pred.tolist()):
                by_exposure[e].append(p)
    s_all = torch.tensor([e["s"] for e in examples])
    summary = {
        "wall_seconds": wall,
        "mean_step_seconds": wall / len(batches),
        "tokens_per_second": total_tokens / wall,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        "final_l_aux_eval_pass": sum(sq_err) / len(sq_err),
        "aux_floor_constant_predictor": float(((s_all - s_all.mean()) ** 2).mean()),
        "head_pred_mean_by_exposure": {str(e): sum(v) / len(v) for e, v in sorted(by_exposure.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_checkpoint(model, out_dir / "checkpoint", meta)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
