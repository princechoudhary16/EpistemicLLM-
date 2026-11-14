import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data import load_rows
from model import build_model

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from common import grading  # noqa: E402  (shared grader, p2_whole_word_v5)
IDK = ("i dont know", "i do not know", "unknown", "cannot determine", "no information")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def grade(answer, reference, question=""):
    return grading.grade(answer, reference, question)[0]


def auroc(labels, scores):
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=str(REPO / "data/synthetic/main_v0/eval.jsonl"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", choices=["dev", "final", "all"], default="dev",
                    help="which eval questions to score; dev is for choosing lambda, final is the one-shot test")
    ap.add_argument("--allow-final", action="store_true",
                    help="required to score the final split (or all), so it is never touched by accident")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-non-cuda", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_non_cuda:
        raise SystemExit("No CUDA device found. Run on the GPU node, or pass --allow-non-cuda for local debugging.")

    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "meta.json").read_text())["config"]
    tokenizer, model = build_model(cfg, device, checkpoint_dir=ckpt, trainable=False)
    model.eval()
    tokenizer.padding_side = "left"

    system_prompt = cfg["system_prompt"]
    if args.split != "dev" and not args.allow_final:
        raise SystemExit("--split final/all scores the held-out test questions. Pass --allow-final only for the final run.")
    rows = load_rows(args.data)
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No eval rows with split={args.split!r} in {args.data}")
    out_rows = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            hidden = model.lm(**enc, position_ids=position_ids, output_hidden_states=True).hidden_states[-1]
            preds = model.head(hidden[:, -1, :]).cpu().tolist()
            gen = model.lm.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        answers = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, pred, ans in zip(chunk, preds, answers):
            ans = ans.strip()
            out_rows.append({**r, "model_answer": ans, "outcome": grade(ans, r["reference_answer"], r["question"]),
                             "grading_version": grading.GRADING_VERSION, "head_pred": pred})

    by_level = defaultdict(list)
    for r in out_rows:
        by_level[r["exposure_level"]].append(r)
    levels = {}
    for e, rs in sorted(by_level.items()):
        n = len(rs)
        levels[str(e)] = {
            "n": n,
            "accuracy": sum(r["outcome"] == "correct" for r in rs) / n,
            "hallucination": sum(r["outcome"] == "incorrect" for r in rs) / n,
            "abstain": sum(r["outcome"] == "abstain" for r in rs) / n,
            "head_pred_mean": float(np.mean([r["head_pred"] for r in rs])),
            "head_pred_std": float(np.std([r["head_pred"] for r in rs])),
        }

    exposure = np.array([r["exposure_level"] for r in out_rows])
    pred = np.array([r["head_pred"] for r in out_rows])
    correct = np.array([r["outcome"] == "correct" for r in out_rows])
    mask = (exposure >= 5) | (exposure <= 2)
    summary = {
        "n_questions": len(out_rows),
        "by_exposure": levels,
        "spearman_head_vs_exposure": float(spearmanr(pred, exposure).correlation),
        "auroc_head_exposure_ge5_vs_le2": auroc(exposure[mask] >= 5, pred[mask]),
        "auroc_head_predicts_correct": auroc(correct, pred),
        "grading": f"{grading.GRADING_VERSION} (common/grading.py)",
        "checkpoint": str(ckpt),
        "split": args.split,
    }
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / f"eval_{Path(args.data).parent.name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in out_rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
