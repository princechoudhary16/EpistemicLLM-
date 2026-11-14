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

import random

from data import collate, encode_aux_row, encode_row, load_rows, select_aux_rows, set_aux_label
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
    ap.add_argument("--aux-zero", default=None,
                    help="answer-masked prompts for 0x facts (P1's aux_zero_prompts.jsonl): extra rows for the head "
                         "only, spread evenly over the steps; no language-model loss")
    ap.add_argument("--aux-target", choices=["exposure", "k"], default="exposure",
                    help="head target: s(exposure) (default) or the fact's K from --k-labels (K(q) experiment)")
    ap.add_argument("--k-labels", default=None, help="JSON written by kq_labels.py (needed for --aux-target k)")
    ap.add_argument("--aux-label-split", choices=["dev", "final"], default=None,
                    help="aux loss only on answer-training rows of this split's facts; the other rows add language-model "
                         "loss only (fact-disjoint labelling)")
    ap.add_argument("--aux-zero-split", choices=["dev", "final"], default=None,
                    help="use only the aux prompts of this split's 0x facts (name-recognition control)")
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
    if args.aux_target == "k" and not args.k_labels:
        raise SystemExit("--aux-target k needs --k-labels")
    if args.aux_zero and (args.aux_target != "exposure" or args.aux_label_split):
        raise SystemExit("--aux-zero cannot be combined with the K(q) / fact-disjoint labelling options")
    k_doc = json.loads(Path(args.k_labels).read_text()) if args.k_labels else None
    k_labels = k_doc["labels"] if (k_doc and args.aux_target == "k") else None
    examples = [
        set_aux_label(
            encode_row(tokenizer, system_prompt, r, cfg["data"]["max_seq_len"], cfg["head"]["max_exposure"]),
            r, args.aux_label_split, k_labels,
        )
        for r in rows
    ]
    n_labelled = int(sum(e.get("aux_mask", 1.0) for e in examples))
    bs = cfg["train"]["batch_size"]
    lm_chunks = [examples[i : i + bs] for i in range(0, len(examples), bs)]

    # Optional aux-only rows. The answer-training chunks (and their order) are unchanged; aux row k is appended to
    # step floor(k * n_steps / n_aux), so they are spread evenly. Shuffled with the training seed so one fact's four
    # wordings are not adjacent. Rows without any label add nothing to the language-model loss.
    aux_examples, aux_path = [], Path(args.aux_zero) if args.aux_zero else None
    if aux_path is not None:
        aux_rows = select_aux_rows(load_rows(aux_path), args.aux_zero_split)
        random.Random(cfg["train"]["seed"]).shuffle(aux_rows)
        aux_examples = [
            encode_aux_row(tokenizer, system_prompt, r, cfg["data"]["max_seq_len"], cfg["head"]["max_exposure"])
            for r in aux_rows
        ]
        for k, ex in enumerate(aux_examples):
            lm_chunks[k * len(lm_chunks) // len(aux_examples)].append(ex)
    batches = [collate(chunk, tokenizer.pad_token_id) for chunk in lm_chunks]

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
        "aux_zero_path": str(aux_path) if aux_path else None,
        "aux_zero_sha256": sha256(aux_path) if aux_path else None,
        "n_aux_rows": len(aux_examples),
        "aux_zero_split": args.aux_zero_split,
        "aux_target": args.aux_target,
        "k_labels_path": args.k_labels,
        "k_labels_sha256": sha256(args.k_labels) if args.k_labels else None,
        "aux_label_split": args.aux_label_split,
        "n_aux_labelled_rows": n_labelled,
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
        "target": ("log(1+e)/log(1+max_exposure)" if args.aux_target == "exposure"
                   else "K(fact) from teacher labels (correct sampled responses / sampled responses)"),
        "aux_label_split": args.aux_label_split,
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
            labelled = b["aux_mask"].cpu() > 0                      # unlabelled rows have no aux target
            sq_err += (((pred - target) ** 2)[labelled]).tolist()
            for e, p in zip(b["exposure"].cpu().tolist(), pred.tolist()):
                by_exposure[e].append(p)
    s_all = torch.tensor([e["s"] for e in examples + aux_examples if e.get("aux_mask", 1.0) > 0])
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
