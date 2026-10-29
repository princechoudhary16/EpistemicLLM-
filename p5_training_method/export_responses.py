"""Export P5 responses in the shared schema (common/schema.py) so P6's compare_methods.py can score
P5 beside base / SFT / IDK:

    python -m p6_baselines.compare_methods --responses base=... sft=... p5=results/raw/p5_main_v0_lam3_seed0.jsonl

Generation and grading go through P6's own ModelRunner and run_eval (same prompt, decoding config and
placeholder grader as every other method). Per (lambda, seed) checkpoint this writes one file,
p5_<ds>_lam<L>_seed<S>.jsonl, where
  confidence_score       = average answer log-prob (the signal every method has; kept comparable, as P6 asked)
  aux_confidence_score   = the evidence head's prediction for the question (one value per question);
                           null for lambda=0, which has no trained head
lambda=0 is plain SFT trained through P5's own code path (a matched baseline for the lambda>0 model).
`aux_confidence_score` is the field P6 added to the schema on branch `sahana` (not yet merged to main).
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from common.evaluate import run_eval  # noqa: E402
from common.model import ModelRunner  # noqa: E402
from common.schema import read_jsonl, write_jsonl  # noqa: E402
from head import EvidenceHead  # noqa: E402


def load_yaml(name):
    with open(REPO / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def head_scores(runner, head, questions, system_prompt, batch_size, device):
    """Head prediction from the final-layer hidden state at the last prompt token (left-padded batches)."""
    tok = runner.tokenizer
    old_side, tok.padding_side = tok.padding_side, "left"
    out = []
    for i in range(0, len(questions), batch_size):
        prompts = [runner._build_prompt(q, system_prompt) for q in questions[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        hidden = runner.model(**enc, position_ids=pos, output_hidden_states=True).hidden_states[-1][:, -1, :]
        out.extend(head(hidden).cpu().tolist())
    tok.padding_side = old_side
    return out


def finalize_rows(rows, head_by_question):
    """Keep confidence_score = average log-prob (the signal every method has; P6 keeps it comparable) and
    put the evidence head's score, one value per question, in aux_confidence_score (null for lambda=0)."""
    method = "p5" if head_by_question is not None else "sft_p5code"
    return [
        dict(r, method=method,
             aux_confidence_score=None if head_by_question is None else head_by_question[r["question_id"]])
        for r in rows
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", default="main_v0")
    ap.add_argument("--lams", nargs="+", default=["0", "3"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--root", default="results/raw/p5", help="where the P5 training runs live")
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: first N questions (not comparable)")
    args = ap.parse_args()

    model_cfg, dec_cfg = load_yaml("model.yaml"), load_yaml("decoding.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.dataset_version
    questions = read_jsonl(REPO / "data" / "synthetic" / ds / "eval.jsonl")[: args.limit]
    qids = [q["question_id"] for q in questions]
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(questions)} questions, decoding={dec_cfg['decoding_config_id']}, prompt={dec_cfg['prompt_version']}, "
          f"dtype={model_cfg['torch_dtype']}", file=sys.stderr, flush=True)

    for seed in args.seeds:
        for lam in args.lams:
            ckpt = Path(args.root) / f"{ds}_lam{lam}_seed{seed}" / "checkpoint"
            if not (ckpt / "adapter").is_dir():
                raise SystemExit(f"missing checkpoint: {ckpt}")
            lam_positive = float(lam) > 0
            print(f"=== lambda={lam} seed={seed}: generating from {ckpt}", file=sys.stderr, flush=True)
            runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], adapter_path=str(ckpt / "adapter"),
                                  torch_dtype=model_cfg["torch_dtype"], device=device)
            rows = run_eval(runner, questions, dec_cfg, model_cfg, f"p5_{ds}_lam{lam}_seed{seed}",
                             "p5", f"{ds}_p5_lam{lam}_seed{seed}", seed=args.gen_seed, training_seed=seed)
            head_by_question = None
            if lam_positive:
                head = EvidenceHead(runner.model.config.hidden_size).to(device)
                head.load_state_dict(torch.load(ckpt / "head.pt", map_location=device))
                head.eval()
                scores = head_scores(runner, head, [q["question"] for q in questions], dec_cfg["system_prompt"],
                                     args.batch_size, device)
                head_by_question = dict(zip(qids, scores))
            stem = f"p5_{ds}_lam{lam}_seed{seed}"
            write_jsonl(finalize_rows(rows, head_by_question), out_dir / f"{stem}.jsonl")
            print(f"    wrote {stem}.jsonl to {out_dir}", file=sys.stderr, flush=True)
            del runner
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
