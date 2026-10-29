"""Control experiment: is the auxiliary loss needed, or would a plain probe do as well?

For each seed, with the plain-SFT (lambda=0) checkpoint loaded, trains the SAME head architecture on
the SAME target s(e) and the SAME training prompts as the P5 head, but on FROZEN hidden states:
  - probe_sft:  hidden states of the plain-SFT model
  - probe_base: hidden states of the original base model (LoRA switched off)
  - probe_on_p5_model: hidden states of the aux-trained (lambda=P5) model, to test whether the
    auxiliary loss makes the representation easier to read than plain SFT does
and compares them with the P5 head (aux-trained, from its saved predictions), the SFT model's average
answer log-probability (P6's current confidence), and the untrained lambda=0 head (no-skill floor).

Probe settings are fixed in advance (Adam, lr 1e-3, 1000 full-batch steps), not tuned on any eval set.
Probes are fact-disjoint: scoring the final split fits them on dev facts only (and vice versa), with an
assertion that no fact is on both sides. The P5 head and the models themselves train on all facts (the
experiment's design), so only the probes are changed by this.
"predicts correct" uses the answers of the model each signal belongs to: the P5 head with the
lambda=P5 model's answers, every other signal with the plain-SFT model's answers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data import evidence_target, load_rows
from head import EvidenceHead
from model import build_model

REPO = Path(__file__).resolve().parent.parent


def chat_prompt(tokenizer, system_prompt, question):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.no_grad()
def last_hidden(lm, tokenizer, system_prompt, questions, bs, device):
    """Final-layer hidden state at the last prompt token (what the head reads)."""
    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(questions), bs):
        prompts = [chat_prompt(tokenizer, system_prompt, q) for q in questions[i : i + bs]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        h = lm(**enc, position_ids=pos, output_hidden_states=True).hidden_states[-1][:, -1, :]
        out.append(h.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def answer_logprob(lm, tokenizer, system_prompt, questions, answers, bs, device):
    """Mean token log-prob of the given answer (EOS excluded), teacher-forced."""
    scores = []
    for i in range(0, len(questions), bs):
        seqs, plens = [], []
        for q, a in zip(questions[i : i + bs], answers[i : i + bs]):
            p = tokenizer(chat_prompt(tokenizer, system_prompt, q), add_special_tokens=False)["input_ids"]
            ans = tokenizer(a, add_special_tokens=False)["input_ids"]
            seqs.append(p + ans)
            plens.append((len(p), len(ans)))
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        ids, mask = ids.to(device), mask.to(device)
        logp = F.log_softmax(lm(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
        for j, (pl, al) in enumerate(plens):
            if al == 0:
                scores.append(-50.0)  # empty answer: lowest confidence
                continue
            tok = ids[j, pl : pl + al]
            scores.append(logp[j, pl - 1 : pl + al - 1].gather(-1, tok[:, None]).mean().item())
    return np.array(scores)


def train_probe(feats, targets, seed, steps, lr, device):
    torch.manual_seed(seed)
    head = EvidenceHead(feats.size(1)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    x, y = feats.to(device), targets.to(device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(head(x), y)
        loss.backward()
        opt.step()
    return head, loss.item()


@torch.no_grad()
def probe_scores(head, feats, device):
    return head(feats.to(device)).cpu().numpy()


def auroc(labels, scores):
    labels = np.asarray(labels)
    return None if labels.min() == labels.max() else float(roc_auc_score(labels, scores))


def signal_metrics(scores, exposure, correct):
    mask = (exposure >= 5) | (exposure <= 2)
    return {
        "spearman": float(spearmanr(scores, exposure).correlation),
        "auroc_ge5_vs_le2": auroc(exposure[mask] >= 5, scores[mask]),
        "auroc_correct": auroc(correct, scores),
    }


def read_preds(root, run_id, dset, split):
    path = Path(root) / run_id / f"eval_{dset}_{split}" / "predictions.jsonl"
    return {r["question_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "final"], default="dev")
    ap.add_argument("--allow-final", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--p5-lambda", default="3")
    ap.add_argument("--dataset", default="main_v0")
    ap.add_argument("--root", default="results/raw/p5")
    ap.add_argument("--probe-steps", type=int, default=1000)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "final" and not args.allow_final:
        raise SystemExit("Pass --allow-final to score the final split.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = REPO / "data/synthetic" / args.dataset
    # Fact-disjoint probes: to score on split S, fit only on training rows of the OTHER split's facts,
    # so a probe cannot memorise the exposure label of any fact it is later tested on.
    fact_split = {f["fact_id"]: f["split"] for f in load_rows(data_dir / "facts.jsonl")}
    fit_split = "dev" if args.split == "final" else "final"
    train_rows = [r for r in load_rows(data_dir / "train.jsonl") if fact_split[r["fact_id"]] == fit_split]
    train_q = [r["prompt"] for r in train_rows]
    probe_facts = {r["fact_id"] for r in train_rows}
    print(f"probes fit on {len(train_rows)} training rows of {len(probe_facts)} {fit_split} facts; "
          f"scored on {args.split} facts", flush=True)

    per_seed = {}
    score_rows = []
    for seed in args.seeds:
        run0 = f"{args.dataset}_lam0_seed{seed}"
        run_p5 = f"{args.dataset}_lam{args.p5_lambda}_seed{seed}"
        preds0 = read_preds(args.root, run0, args.dataset, args.split)
        preds5 = read_preds(args.root, run_p5, args.dataset, args.split)
        qids = list(preds0)
        eval_facts = {preds0[q]["fact_id"] for q in qids}
        assert not (eval_facts & probe_facts), "probe fit facts overlap the facts it is scored on"
        assert all(fact_split[f] == args.split for f in eval_facts), "eval rows are not all from the requested split"
        exposure = np.array([preds0[q]["exposure_level"] for q in qids])
        questions = [preds0[q]["question"] for q in qids]
        answers = [preds0[q]["model_answer"] for q in qids]
        correct0 = np.array([preds0[q]["outcome"] == "correct" for q in qids])
        correct5 = np.array([preds5[q]["outcome"] == "correct" for q in qids])

        ckpt = Path(args.root) / run0 / "checkpoint"
        cfg = json.loads((ckpt / "meta.json").read_text())["config"]
        system_prompt = cfg["system_prompt"]
        max_e = cfg["head"]["max_exposure"]
        targets = torch.tensor([evidence_target(r["exposure_level"], max_e) for r in train_rows], dtype=torch.float32)

        tokenizer, model = build_model(cfg, device, checkpoint_dir=ckpt, trainable=False)
        lm = model.lm.eval()
        print(f"[seed {seed}] extracting features", flush=True)
        sft_train = last_hidden(lm, tokenizer, system_prompt, train_q, args.batch_size, device)
        sft_eval = last_hidden(lm, tokenizer, system_prompt, questions, args.batch_size, device)
        logp = answer_logprob(lm, tokenizer, system_prompt, questions, answers, args.batch_size, device)
        with lm.disable_adapter():
            base_train = last_hidden(lm, tokenizer, system_prompt, train_q, args.batch_size, device)
            base_eval = last_hidden(lm, tokenizer, system_prompt, questions, args.batch_size, device)

        del model, lm
        torch.cuda.empty_cache()

        # Same kind of probe on the frozen hidden states of the aux-trained (P5) model:
        # does the auxiliary loss make the representation easier to read than plain SFT does?
        ckpt5 = Path(args.root) / run_p5 / "checkpoint"
        cfg5 = json.loads((ckpt5 / "meta.json").read_text())["config"]
        _, model5 = build_model(cfg5, device, checkpoint_dir=ckpt5, trainable=False)
        lm5 = model5.lm.eval()
        p5_train = last_hidden(lm5, tokenizer, system_prompt, train_q, args.batch_size, device)
        p5_eval = last_hidden(lm5, tokenizer, system_prompt, questions, args.batch_size, device)
        answers5 = [preds5[q]["model_answer"] for q in qids]
        logp5 = answer_logprob(lm5, tokenizer, system_prompt, questions, answers5, args.batch_size, device)
        del model5, lm5
        torch.cuda.empty_cache()

        head_sft, loss_sft = train_probe(sft_train, targets, seed, args.probe_steps, args.probe_lr, device)
        head_base, loss_base = train_probe(base_train, targets, seed, args.probe_steps, args.probe_lr, device)
        head_p5, loss_p5 = train_probe(p5_train, targets, seed, args.probe_steps, args.probe_lr, device)
        print(f"[seed {seed}] probe train MSE: sft {loss_sft:.4f}, base {loss_base:.4f}, p5 {loss_p5:.4f}", flush=True)

        # (scores, outcomes of the model the signal belongs to)
        signals = {
            "p5_head": (np.array([preds5[q]["head_pred"] for q in qids]), correct5),
            "probe_on_sft": (probe_scores(head_sft, sft_eval, device), correct0),
            "probe_on_p5_model": (probe_scores(head_p5, p5_eval, device), correct5),
            "probe_on_base": (probe_scores(head_base, base_eval, device), correct0),
            "sft_avg_logprob": (logp, correct0),
            "p5_avg_logprob": (logp5, correct5),
            "untrained_head_lam0": (np.array([preds0[q]["head_pred"] for q in qids]), correct0),
        }
        per_seed[seed] = {name: signal_metrics(s, exposure, c) for name, (s, c) in signals.items()}
        per_seed[seed]["probe_train_mse"] = {"sft": loss_sft, "base": loss_base, "p5": loss_p5}
        for i, q in enumerate(qids):
            score_rows.append({
                "seed": seed, "question_id": q, "exposure_level": int(exposure[i]),
                "outcome_sft": preds0[q]["outcome"], "outcome_p5": preds5[q]["outcome"],
                "scores": {name: float(s[i]) for name, (s, _) in signals.items()},
            })

    names = ["p5_head", "probe_on_p5_model", "probe_on_sft", "probe_on_base",
             "sft_avg_logprob", "p5_avg_logprob", "untrained_head_lam0"]
    keys = ["spearman", "auroc_ge5_vs_le2", "auroc_correct"]
    table = {}
    print(f"\nsplit={args.split}, seeds={args.seeds}, P5 lambda={args.p5_lambda}")
    print(f"{'signal':<22}" + "".join(f"{k:>24}" for k in keys))
    for name in names:
        table[name] = {}
        cells = []
        for k in keys:
            vals = [per_seed[s][name][k] for s in args.seeds]
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            table[name][k] = [m, sd]
            cells.append(f"{m:>16.3f} +-{sd:.3f}")
        print(f"{name:<22}" + "".join(f"{c:>24}" for c in cells))

    out = Path(args.out or f"results/final/p5_probe_controls_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": table, "per_seed": per_seed, "probe_steps": args.probe_steps,
                               "probe_lr": args.probe_lr, "probe_fit_facts": fit_split,
                               "n_probe_fit_rows": len(train_rows)}, indent=2))
    # name follows --out so another dataset's results never overwrite these:
    # p5_probe_controls_final.json -> p5_signal_scores_final.jsonl, p5_main_w1_probe_controls_final.json -> p5_main_w1_signal_scores_final.jsonl
    scores_out = out.with_name(out.name.replace("probe_controls", "signal_scores").replace(".json", ".jsonl"))
    assert scores_out != out, "--out must contain 'probe_controls' so the scores file gets its own name"
    scores_out.write_text("\n".join(json.dumps(r) for r in score_rows))
    print(f"wrote {out} and {scores_out}")


if __name__ == "__main__":
    main()
